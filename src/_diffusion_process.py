# Created by Javad Komijani, 2024

import torch
import scipy
import numpy as np

Tensor = torch.Tensor


# =============================================================================
class DiffusionProcess(torch.nn.Module):
    r"""Implements a diffusion process."""

    def __init__(self, score_func: torch.nn.Module, omega: float = np.pi):

        super().__init__()
        self.score_func = score_func
        self.omega = omega
        self.sigma = (2 * omega)**0.5  # hard-wired

    def forward_drift(self, t: Tensor, x_t: Tensor):
        """Computes the drift term in the diffusion process."""
        return - self.omega * x_t

    def reverse_drift(self, t: Tensor, x_t: Tensor, sigma_tilde: float = 0):
        """
        Computes the drift term in the reverse diffusion process.

        Args:
            - t (Tensor): A 1-dimensional tensor representing time,
              corresponding to the batch axis of `x_t`.
            - x_t (Tensor): The state of the system at time `t`.
            - sigma_tilde (float, optional): A scaling factor for noise
              adjustment. Default is `0`.

        Returns:
            Tensor: The computed drift term for the reverse diffusion process.
        """
        if t.shape[0] == 1:
            t = t.repeat(x_t.shape[0])
        score = self.score_func(t, x_t)
        coeff = (self.omega + sigma_tilde**2 / 2)

        return -self.omega * x_t - coeff * score

    def run_diffusion_process_for_training(
        self, x_0: Tensor, n_steps: Tensor, step_size: float
    ):
        """Simulates the (forward) diffusion processes."""
  
        time = step_size * n_steps.reshape(-1, *[1] * (x_0.ndim - 1))
        dcy = self.decay_factor(time)
        std = self.effective_noise_std(0, time)
        eps = torch.randn_like(x_0)

        x_t = dcy * x_0 + std * eps

        return x_t, eps, std

    def run_diffusion_process(self, x_init, t_init=0, t_eval=[1]):
        """Simulates the (forward) diffusion processes."""

        batch_size = x_init.shape[0]

        t_shape = [-1, *[1] * (x_init.ndim - 1)]
        t_init = torch.tensor(t_init).reshape(*t_shape)

        x_eval = [None] * len(t_eval)

        for ind in range(len(t_eval)):

            t = torch.tensor(t_eval[ind]).reshape(*t_shape)

            dcy = self.decay_factor(t - t_init)
            std = self.effective_noise_std(t_init, t)
            eps = torch.randn_like(x_init)

            x_eval[ind] = dcy * x_init + std * eps

            # for the next round
            t_init = t
            x_init = x_eval[ind]

        return x_eval

    def run_denoising_process(
        self, x_init, t_init=1, t_eval=[0], sigma_tilde=None
    ):
        # UNDER CONSTRUCTION
        """Simulates the reverse diffusion process."""

        if sigma_tilde is None:
            sigma_tilde = self.sigma

    def decay_factor(self, t):
        r"""Returns :math:`e^{-\omega t}`."""
        return torch.exp(- self.omega * t)

    def effective_noise_std(self, t_a, t_b):
        r"""
        Returns the standard deviation of the effective noise at time `t` in
        the closed-form solution of the diffusion process with vanishing drift
        in variance-expanding picture.

        This method computes the normalizaed root-mean-square (RMS) of
        :math:`g(t) = e^{\omega t} \sqrt{2 \omega}` using the formula:

        .. math::

            \text{RMS} = \sqrt{| \int_a^b dt g(t)^2 |} e^{-\omega b}

        The output represents the coefficient of the normal noise term in the
        SDE, determining the standard deviation of the noise contributing to
        the SDE. The function utilizes the closed-form solution
        math:`\sqrt{e^{-2 \omega t_a} - e^{-2 \omega t_b}}`.
        """
        coeff = 2 * self.omega
        return (1 - torch.exp(coeff * (t_a - t_b))).abs()**0.5


# =============================================================================
class DenoisingFlow:

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        solver: str = 'scipy.solve_ivp'
    ):

        super().__init__()

        self.diffusion_process = diffusion_process

        if solver == 'scipy.solve_ivp':
            solver = self.scipy_solve_ivp
        else:
            assert not isinstance(str, solver), "solver not recognized"

        self.solver = solver

    def forward(self, x_init, t_span=[1, 0], t_eval=[0], **solver_kwargs):
        return self.solver(t_span, x_init, t_eval=t_eval, **solver_kwargs)

    def reverse(self, x_init, t_span=[0, 1], t_eval=[1], **solver_kwargs):
        return self.solver(t_span, x_init, t_eval=t_eval, **solver_kwargs)

    @torch.no_grad
    def scipy_solve_ivp(self, t_span, x_init, t_eval=[0], **solve_kwargs):

        score_func = self.diffusion_process.reverse_drift

        x_shape = x_init.shape
        x_init = x_init.cpu().numpy().ravel()

        def scipy_wrapper(t, x):
            return score_func(
                    torch.tensor([t]), torch.tensor(x.reshape(x_shape))
                    ).cpu().numpy().ravel()

        res = scipy.integrate.solve_ivp(
            scipy_wrapper, t_span, x_init, t_eval=t_eval, **solve_kwargs
        )
        if len(t_eval) == 1:
            return torch.tensor(res.y).reshape(*x_shape)
        else:
            return torch.tensor(res.y).reshape(*x_shape, len(t_eval))

    def torch_solve_ivp(self, simga_tilde=0, **kwargs):
        # UNDER CONSTRUCTION

        drift = Drift4ReverseProcess(
            self.score_func,
            omega=self.omega,
            sigma_tilde=sigma_tilde
        )

        kwargs.update({'t_span': [1, 0]})

        if sigma_tilde == 0:
            reverse_flow = ODEflow(drift, **kwargs)

        else:
            reverse_flow = StochODEflow(
                drift, noise_std_func=lambda *args: sigma_tilde, **kwargs
            )

        self.reverse_flow = reverse_flow
