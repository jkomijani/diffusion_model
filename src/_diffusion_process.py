# Created by Javad Komijani, 2024

"""Implements the diffusion process."""

import torch
import scipy
import numpy as np

from torch_solve_ext.integrate import ODEflow

Tensor = torch.Tensor


# =============================================================================
class DiffusionProcess:
    r"""Implements a diffusion process as:

    .. math:

        d x(t) / dt = - \omega x(t) + \sigma \eta(t)

    with :math:`\sigma = \sqrt{2\omega}`.
    """

    def __init__(self, score_func: torch.nn.Module, omega: float = np.pi):

        self.score_func = score_func
        self.omega = omega
        self.sigma = (2 * omega)**0.5  # hard-wired

    def denoising_drift(self, t: Tensor, x_t: Tensor, sigma_tilde: float = 0):
        """
        Computes the drift term in the denoising process.

        Args:
            - t (Tensor): A 1-dimensional tensor representing time,
              corresponding to the batch axis of `x_t`.
            - x_t (Tensor): The state of the system at time `t`.
            - sigma_tilde (float, optional): A scaling factor for noise
              adjustment. Default is `0`.

        Returns:
            Tensor: The computed drift term for the reverse diffusion process.
        """
        score = self.score_func(t, x_t)
        coeff = self.omega + sigma_tilde**2 / 2

        return -self.omega * x_t - coeff * score

    def run_for_training(self, x_0: Tensor, t_steps: Tensor, step_size: float):
        """
        Suitable for training: simulates the diffusion processes from 0 to
        `t_steps`, which is a 1 dimensional tensor.
        """
        # first we increase the dimensions of `t_steps` to match `x_0`
        t_steps = t_steps.view(-1, *[1] * (x_0.ndim - 1))
        t_eval = step_size * t_steps

        dcy = torch.exp(- self.omega * t_eval)  # decay factor of signal
        std = torch.sqrt(1 - dcy**2)  # std of integrated noise at time t_eval
        eps = torch.randn_like(x_0)  # normalized noise

        x_t = dcy * x_0 + std * eps

        return x_t, eps, std

    def run_diffusion_process(self, x_0: Tensor, t_0=0, t_eval=(1,)):
        """Simulates the diffusion processes."""

        x_eval = [None] * len(t_eval)

        for ind, t in enumerate(t_eval):

            assert t >= t_0, "`t_eval` must monotonically increase."

            dcy = np.exp(- self.omega * (t - t_0))  # signal's decay factor
            std = np.sqrt(1 - dcy**2)  # std of integrated noise at time t
            eps = torch.randn_like(x_0)  # normalized noise

            x_eval[ind] = dcy * x_0 + std * eps

            x_0, t_0 = x_eval[ind], t  # for the next round

        return x_eval

    def run_denoising_process(self, x_0, t_0=1, t_eval=(0,), sigma_tilde=None):
        """Simulates the denoising process."""
        # ***UNDER CONSTRUCTION***
        if sigma_tilde is None:
            sigma_tilde = self.sigma

    def denoising_flow(self, t_span=(1, 0), method='Euler', step_size=0.001):
        """
        Creates and returns an instance of `ODEflow` that performs denoising
        using a deterministic ordinary differential equation (ODE).

        Unlike `run_denoising_process`, which solves a stochastic differential
        equation (SDE) for denoising, this method follows a deterministic
        approach.

        Note that the `ODEflow` class has `forward` and `reverse` methods.
        """
        kwargs = {'method': method, 'step_size': step_size}
        return ODEflow(self.denoising_drift, t_span, **kwargs)


# =============================================================================
@torch.no_grad
def scipy_solve_ivp(func, t_span, x_0, t_eval=(0,), **solver_kwargs):
    """Solving an initial value problem with scipy."""
    x_shape = x_0.shape
    x_0 = x_0.cpu().numpy().ravel()

    def func_scipy_wrapper(t, x):
        t = torch.tensor([t]).repeat(x_shape[0])
        x = torch.tensor(x.reshape(x_shape))
        return func(t, x).cpu().numpy().ravel()

    res = scipy.integrate.solve_ivp(
        func_scipy_wrapper, t_span, x_0, t_eval=t_eval, **solver_kwargs
    )

    y_reshape = x_shape if len(t_eval) == 1 else (*x_shape, len(t_eval))

    return torch.tensor(res.y).reshape(*y_reshape)
