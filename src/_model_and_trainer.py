# Created by Javad Komijani, 2024

"""
This module contains high-level classes for diffusion-based generating models,
with the central `Model` class integrating essential components such as priors,
diffusion processes, and actions. It provides utilities for training and
sampling, along with support for MCMC sampling and device management.
"""

import time
import torch

from .device import ModelDeviceHandler


# =============================================================================
class Model:
    """
    The central high-level class of the package, which integrates instances of
    essential classes (`prior`, `diffusion_process`, and `action`) to provide
    utilities for training and sampling. This class interfaces with various
    core components to facilitate training, posterior inference, MCMC
    sampling, and device management.
    """
    def __init__(self, *, prior, diffusion_process, action):

        self.prior = prior
        self.action = action

        self.diffusion_process = diffusion_process

        self.train = Trainer(self)

        self.device_handler = ModelDeviceHandler(self)

    @property
    def denoising_flow(self):
        """
        Denoising flow is the deterministic evolution in the reverse direction.
        """
        return self.diffusion_process.denoising_flow


# =============================================================================
class Trainer:
    """A class for training a given model."""

    optimizer_class = torch.optim.AdamW
    optimizer = None
    scheduler = None

    loss_c0 = None  # c_0 in the augmented loss functions

    def __init__(self, model: Model):

        self._model = model

        # Initialize training history tracking
        self.train_history = {'epoch': 0, 'loss': []}

        # Default hyperparameters
        self.hyperparam = {'fused': False, 'betas': (0.9, 0.99)}

        # Checkpoint configuration
        self.checkpoint_dict = {'print_every': None}

        # Default loss function
        self.loss_func = self.implicit_score_matching

    def __call__(
        self,
        data_loader,
        n_epochs: int = 200,
        n_timesteps: int = 1000,
        optimizer_class=None,
        scheduler=None,
        loss_func=None,
        loss_c0: float = 0,
        hyperparam=None,
        checkpoint_dict=None
    ):
        """Fit the model; i.e. train the model.

        Parameters
        ----------
        data_loader: generator or None
            Loads true samples for training.

        n_epochs: int
            Number of epochs of training.

        n_timesteps: int
            The total number of steps for reaching `t = 1` from `t = 0`.

        optimizer_class : optimization class, optional
            By default is set to torch.optim.AdamW, but can be changed.

        scheduler : scheduler class, optional
            By default no scheduler is used.

        loss_func : None or callable, optional
            The default value is None, which translates to using the implicit
            score matching with the conventional weight over time.

        hyperparam : dict, optional
            Can be used to set hyperparameters like the learning rate and decay
            weights.

        checkpoint_dict : dict, optional
            Can be set to control printing the status of the training.
        """
        # Update the attributes of the instance
        if hyperparam is not None:
            self.hyperparam.update(hyperparam)

        if checkpoint_dict is not None:
            self.checkpoint_dict.update(checkpoint_dict)

        if loss_func is not None:
            self.loss_func = loss_func

        self.loss_c0 = loss_c0

        if optimizer_class is not None:
            self.optimizer_class = optimizer_class

        parameters = self._model.diffusion_process.score_func.parameters()
        self.optimizer = self.optimizer_class(parameters, **self.hyperparam)

        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)

        if n_epochs > 0:
            self._train(data_loader, n_epochs, n_timesteps)

    def _train(self, data_loader, n_epochs, n_timesteps):

        self.train_history['loss'].extend([None] * n_epochs)

        rank = self._model.device_handler.rank

        last_epoch = self.train_history['epoch']
        report_progress = self.checkpoint_dict['print_every'] is not None

        if rank == 0 and report_progress:
            print(f">>> Training started for {n_epochs} epochs <<<")

        t_1 = time.time()

        for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):

            loss = self.step(data_loader, n_timesteps)

            self._checkpoint(epoch, loss)

            if self.scheduler is not None:
                self.scheduler.step()

        t_2 = time.time()

        if rank == 0 and report_progress:
            print(f">>> Training finished ({loss.device});", end='')
            print(f" TIME = {t_2 - t_1:.3g} sec <<<")

    def step(self, data_loader, n_steps):
        """Perform a train step."""

        process = self._model.diffusion_process
        action = self._model.action

        loss_sum = 0
        n_samples = 0
        dt = 1 / n_steps

        for x_0, in data_loader:

            bsize = x_0.shape[0]

            t_steps = 1 + torch.randint(n_steps, (bsize,), device=x_0.device)

            x_t, eps, noise_std = process.run_for_training(x_0, t_steps, dt)

            score = process.score_func(dt * t_steps, x_t)

            loss = self.loss_func(score, eps, noise_std)

            if self.loss_c0 > 0:
                s_0 = process.score_func(0 * t_steps, x_0)  # score at time 0
                loss += self.loss_c0 * torch.mean((s_0 - action.force(x_0))**2)

            self.optimizer.zero_grad()  # clears old gradients from last steps
            loss.backward()
            self.optimizer.step()

            loss_sum += loss * bsize
            n_samples += bsize

        return loss_sum / n_samples

    @staticmethod
    def implicit_score_matching(score, eps, noise_std):
        r"""
        Returns the loss based on implicit score matching with the weight over
        time set to the variance of the effective noise added to the variable.
        """
        res = score * noise_std + eps
        return torch.mean(res * res.conj()).real

    @torch.no_grad()
    def _checkpoint(self, epoch, loss):

        rank = self._model.device_handler.rank

        every = self.checkpoint_dict['print_every']

        loss = self._model.device_handler.all_gather_into_tensor(loss).item()

        if rank == 0 and every is not None and epoch % every == 0:
            print(f"Epoch: {epoch} | loss: {loss:.4f}")

        if rank == 0:
            self.train_history['epoch'] = epoch
            self.train_history['loss'][epoch - 1] = loss
