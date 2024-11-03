# Copyright (c) 2024 Javad Komijani


import torch
import numpy as np
import time

from ._diffusion_process import DiffusionProcess 
from ._diffusion_process import DenoisingFlow

from .device import ModelDeviceHandler


# =============================================================================
class Model:

    def __init__(self, *, prior, action, diffusion_process):

        self.prior = prior
        self.action = action

        self.diffusion_process = diffusion_process
        self.denoising_flow = DenoisingFlow(diffusion_process)

        self.train = Fitter(self)

        self.device_handler = ModelDeviceHandler(self)
        self._net = self.diffusion_process  # for use in self.device_handler


# =============================================================================
class Fitter:
    """A class for training a given model."""

    def __init__(self, model: Model):
        self._model = model

        self.train_history = dict(loss=[None])

        self.hyperparam = dict(lr=0.01, weight_decay=0.01, betas=(0.9, 0.99))

        self.checkpoint_dict = dict(print_stride=1)

    def __call__(self,
            data_loader,
            n_epochs = 200,
            optimizer_class = torch.optim.AdamW,
            scheduler = None,
            loss_func = None,
            hyperparam = {},
            checkpoint_dict = {}
            ):

        """Fit the model; i.e. train the model.

        Parameters
        ----------
        data_loader: generator or None
            Loads true samples for training.
           
        n_epochs : int
            Number of epochs of training.

        optimizer_class : optimization class, optional
            By default is set to torch.optim.AdamW, but can be changed.

        scheduler : scheduler class, optional
            By default no scheduler is used.

        loss_func : None or callable, optional
            The default value is None, which translates to using the
            conventioanl weight over time.

        hyperparam : dict, optional
            Can be used to set hyperparameters like the learning rate and decay
            weights.

        checkpoint_dict : dict, optional
            Can be set to control printing the status of the training.
        """
        self.hyperparam.update(hyperparam)
        self.checkpoint_dict.update(checkpoint_dict)

        if loss_func is None:
            self.loss_func = self.loss_with_ms_weight
        else:
            self.loss_func = loss_func

        parameters = self._model.diffusion_process.parameters()
        self.optimizer = optimizer_class(parameters, **self.hyperparam)

        if scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer)

        if n_epochs > 0:
            self._train(data_loader, n_epochs)

    def _train(self, data_loader, n_epochs, loss_c0=0):

        initial_epoch = len(self.train_history['loss'])

        self.train_history['loss'].extend([None] * n_epochs)

        rank = self._model.device_handler.rank

        if rank == 0:
            print(f">>> Training started for {n_epochs} epochs <<<")

        t_1 = time.time()

        for epoch in range(initial_epoch, initial_epoch + n_epochs):

            loss = self.step(data_loader, loss_c0=loss_c0)

            self._checkpoint(epoch, loss)

            if self.scheduler is not None:
                self.scheduler.step()

        t_2 = time.time()

        if rank == 0:
            print(f">>> Training finished ({loss.device});", end='')
            print(f" TIME = {t_2 - t_1:.3g} sec <<<")

    def step(self, data_loader, loss_c0=0):
        """Perform a train step with a batch of inputs of size `batch_size`."""

        process = self._model.diffusion_process
        omega = self._model.diffusion_process.omega
        action = self._model.action

        loss_sum = 0
        n_samples = 0

        for x_0, in data_loader:

            batch_size = x_0.shape[0]

            rand_t = torch.rand(batch_size, device=x_0.device)
            reshapeed_rand_t = rand_t.reshape(-1, *[1]* (x_0.ndim - 1))

            decay_factor = process.decay_factor(reshapeed_rand_t)
            rms_noise_std = process.rms_noise_std(reshapeed_rand_t)

            eta = torch.randn_like(x_0)

            x_t = decay_factor * x_0 + rms_noise_std * eta

            score = process.reduced_score_func(rand_t, x_t) - x_t

            loss = self.loss_func(score, rms_noise_std, eta)

            if loss_c0 > 0:
                t_0 = torch.zeros(batch_size, device=x_0.device)
                score_0 = process.reduced_score_func(t_0, x_0) - x_0
                loss += loss_c0 * torch.sum((score_0 - action.force(x_0))**2)

            self.optimizer.zero_grad()  # clears old gradients from last steps
            loss.backward()
            self.optimizer.step()

            loss_sum += loss * batch_size
            n_samples += batch_size

        return loss_sum / n_samples

    def loss_with_ms_weight(self, score, rms_noise_std, eta):
        """The default loss function."""
        return torch.mean((score * rms_noise_std + eta)**2)

    def loss_with_rms_weight(self, score, rms_noise_std, eta):
        """A loss function with an alternative weight over time."""
        omega = self._model.diffusion_process.omega
        const = torch.acosh(torch.exp(omega)) / omega * np.prod(eta.shape)
        return torch.mean(score * (score * rms_noise_std + 2 * eta)) + const

    @torch.no_grad()
    def _checkpoint(self, epoch, loss):

        rank = self._model.device_handler.rank

        stride = self.checkpoint_dict['print_stride']

        loss = self._model.device_handler.all_gather_into_tensor(loss).item()

        if rank == 0 and (epoch % stride == 0 or epoch == 1):
            print(f"Epoch: {epoch} | loss: {loss:.4f}")

        if rank == 0:
            self.train_history['loss'][epoch] = loss
