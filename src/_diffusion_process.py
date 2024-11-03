# Copyright (c) 2024 Javad Komijani


import torch
import scipy
import numpy as np


Tensor = torch.Tensor


# =============================================================================
class DiffusionProcess(torch.nn.Module):

    def __init__(self,
            reduced_score_func = None,
            omega = np.pi,
            omega_requires_grad = False
            ):

        super().__init__()

        self.reduced_score_func = reduced_score_func
 
        self.omega = torch.nn.Parameter(
                torch.tensor(omega), requires_grad=omega_requires_grad
                )

        self._sigma = (2 * self.omega)**0.5  # hard-wired

    def forward_drift(self, t: Tensor, x_t: Tensor):
        """Returns the drift term in the forward diffusion process."""
        return 0

    def reverse_drift(self, t: Tensor, x_t: Tensor, sigma_tilde: float = 0):
        """Returns the drift term in the reverse diffusion process.

        Parameters
        ----------
        t: Tensor
            Zero or one dimensional Tensor.
        """

        if t.shape[0] == 1:
            t = t.repeat(x_t.shape[0])

        rscore = self.reduced_score_func(t, x_t)

        # score_coeff = (sigma**2 + sigma_tilde**2) / 2 with sigma**2 = 2 omega
        score_coeff = self.omega + sigma_tilde**2 / 2

        if sigma_tilde == 0:
            return - score_coeff * rscore
        else:
            return - score_coeff * rscore + (sigma_tilde**2 / 2) * x_t

    def run_diffusion_process(self, x_init, t_init=0, t_eval=[1]):
        """Simulates the (forward) diffusion processes."""

        batch_size = x_init.shape[0]

        x_eval = [None] * len(t_eval)

        t_shape = [-1, *[1]* (x_init.ndim - 1)]

        for ind in range(len(t_eval)):
            
            t = torch.tensor(t_eval[ind]).reshape(*t_shape)

            decay_factor = self.decay_factor(t - t_init)
            rms_noise_std = self.rms_noise_std(t - t_init)

            eta = torch.randn_like(x_init)

            x_eval[ind] = decay_factor * x_init + rms_noise_std * eta

            # for the next round
            t_init = t
            x_init = x_eval[ind]

        return x_eval

    def run_denoising_process(
            self, x_init, t_init=1, t_eval=[0], sigma_tilde=None
            ):
        """Simulates the reverse diffusion processes."""
        # UNDER CONSTRUCTION

        if sigma_tilde is None:
            sigma_tilde = self._sigma

    def decay_factor(self, t):
        r"""Returns :math:`e^{-\omega t}`."""
        return torch.exp(- self.omega * t)

    def rms_noise_std(self, t):
        r"""Returns the normalized root-mean-square (RMS) of the function
        :math:`g(t) = e^{\omega t} \sqrt{2 \omega}`.

        This method computes the normalized RMS using the formula:

        .. math::

            \text{RMS} = \sqrt{| \int_0^t ds g(s)^2 |}  e^{-\omega t}

        The output represents the coefficient of the normal noise term in a
        stochastic differential equation (SDE), determining the standard
        deviation of the noise contributing to the SDE. The function utilizes
        the closed-form solution "math:`\sqrt{1 - e^{-2 \omega t}}`.
        """
        return (1 - torch.exp(-2 * self.omega * t)).abs()**0.5


# =============================================================================
class DenoisingFlow:

    def __init__(self,
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

        rscore_func = self.diffusion_process.reverse_drift

        x_shape = x_init.shape
        x_init = x_init.cpu().numpy().ravel()

        def scipy_wrapper(t, x):
            return rscore_func(
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
                    self.reduced_score_func,
                    omega = self.omega,
                    sigma_tilde = sigma_tilde
                    )

        kwargs.update(dict(t_span = [1, 0]))

        if sigma_tilde == 0:
            reverse_flow = ODEflow(drift, **kwargs)

        else:
            reverse_flow = StochODEflow(
                drift, noise_std_func = lambda *args: sigma_tilde, **kwargs
                )

        self.reverse_flow = reverse_flow
