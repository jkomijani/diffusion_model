# Created by Javad Komijani, 2025

"""Implements a diffusion process governing Lie groups."""

from abc import abstractmethod, ABC

import torch
import numpy as np

from torch_solve_ext.integrate import LieGroupODEflow

Tensor = torch.Tensor


# =============================================================================
class LieGroupDiffusionProcess(ABC):
    r"""
    Implements a diffusion process governing Lie groups.

    The process follows the stochastic differential equation:

    .. math::
        \frac{d y(t)}{dt} = \sigma e^{\omega t} \eta_t

    where :math:`\sigma = \sqrt{2\omega}`, and :math:`\eta_t` represents noise.
    This diffusion process governs the evolution of a state on a Lie group,
    with the time-dependent, variance-expanding noise.
    """

    def __init__(self, score_func: torch.nn.Module, omega: float = np.pi):
        r"""
        Initializes the LieGroupDiffusionProcess with the given parameters.

        Args:
            - score_func (torch.nn.Module): The neural network module used to
              esimated the score function (gradient of the log-probability).
            - omega (float, optional): A constant influencing the rate of
              diffusion or the rate of increament in the noise variance.
              Default is :math:`\pi`.
        """
        self.score_func = score_func
        self.omega = omega
        self.sigma = (2 * omega)**0.5  # Hardwired value

    def denoising_drift(self, t: Tensor, y_t: Tensor, sigma_tilde: float = 0):
        """
        Computes the drift term in the denoising process.

        The drift term governs the direction of the reverse diffusion process,
        helping to denoise the state `y_t` based on the score function and the
        time-dependent coefficients.

        Args:
            - t (Tensor): A 1-dimensional tensor representing the time,
              corresponding to the batch axis of `y_t`.
            - y_t (Tensor): The current state of the system at time `t`.
            - sigma_tilde (float, optional): A scaling factor to adjust the
              noise. Default is `0`, meaning no noise.

        Returns:
            Tensor: The drift term, which is used in the denoising process.
        """
        score = self.score_func(t, y_t)
        omega = self.omega
        coeff = (omega + sigma_tilde**2 / 2) * torch.exp(2 * omega * t)

        return - coeff * score

    def run_for_training(self, y_0: Tensor, t_steps: Tensor, step_size: float):
        """
        Simulates the forward diffusion process for training purposes.

        This method simulates the forward diffusion process starting from an
        initial state `y_0`, which is a tensor of the Lie group elements, over
        discrete time steps `t_steps`. The method computes noisy states and
        their noise characteristics, which can be used for training denoising
        models.

        Args:
            - y_0 (Tensor): The initial state of the system at time `t = 0`.
            - t_steps (Tensor): A 1D tensor of random time steps corresponding
              to the batch axis of `x_0`.
            - step_size (float): The step size used for numerical integration.

        Returns:
            tuple:
                - Tensor: `y_t` (noisy state): The noisy states after applying
                  the diffusion process over the specified time steps.

                - Tensor: `alg / std` (normalized algebraic state): The state
                  in the tangent space of the Lie group, normalized by its
                  standard deviation. This tensor shows how noise evolves in
                  the algebraic space over time.

                - Tensor: `std`: The accumulated standard deviation (noise)
                  over time. This tensor tracks how much noise has been added
                  to the state during the diffusion process.
        """
        assert t_steps.shape[0] == y_0.shape[0]

        max_t_steps = torch.max(t_steps)
        t_steps = t_steps.view(-1, *[1] * (y_0.ndim - 1))

        t = 0  # Initial time set to 0
        w = self.omega  # Diffusion rate parameter

        # Initialize algebraic state and variance tensors
        alg = torch.zeros_like(y_0)  # Random walk in `algebra` space
        var = torch.zeros_like(t_steps + 0.)  # Variance of alg at each time

        # Loop over the maximum number of steps
        for step in range(1, 1 + max_t_steps):

            t += step_size

            # Calculate standard deviation for noise and generate noise
            std = (1 - np.exp(-2 * w * step_size))**0.5 * np.exp(w * t)
            # Make std is 0 if time is larger than the evaluation time
            std = std * (step <= t_steps)

            randn_grp, randn_alg = self.randn_lie_like(y_0.real, scale=std)

            y_0 = randn_grp @ y_0
            alg = randn_alg + alg
            var += std**2

        return y_0, alg / var**0.5, var**0.5

    def run_diffusion_process(self, y_0, t_0=0, t_eval=(1,), step_size=0.001):
        """Simulates the diffusion process over specified times.

        This method simulates the forward diffusion process starting from an
        initial state `y_0`. The diffusion is applied till the specified times
        given by `t_eval`, updating the state incrementally at each time step.
        The method returns the states at the specified evaluation times.

        Args:
            - y_0 (Tensor): The initial state of the system at time `t = 0`.
            - t_0 (float, optional): The starting time. Default is `0`.
            - t_eval (tuple of flaots): A tuple of evaluation times at which to
              observe the state. Each time in `t_eval` should be greater than
              or equal to `t_0` and it must be sorted from small to large.
            - step_size (float, optional): The step size for numerical
              integration. Default is `0.001`.

        Returns:
            list:
                - A list of states (`y_eval`) at the specified evaluation
                  times in `t_eval`.
        """
        y_eval = [None] * len(t_eval)  # Initialize list for evaluated states
        t = t_0  # Current time set to t_0
        w = self.omega  # Diffusion rate parameter

        # Iterate through evaluation times
        for ind, t_target in enumerate(t_eval):

            assert t <= t_target, "`t_eval` must monotonically increase."

            while t < t_target:  # Iterate until reaching the target time

                t += step_size

                # Calculate standard deviation for noise and generate noise
                std = (1 - np.exp(-2 * w * step_size))**0.5 * np.exp(w * t)
                randn_grp, _ = self.randn_lie_like(y_0.real, scale=std)

                # Update the state and time
                y_0 = randn_grp @ y_0

            y_eval[ind] = y_0  # Store the state at the current evaluation time

        return y_eval  # Return the list of evaluated states

    def run_denoising_process(self, y_0, t_0=1, t_eval=(0,), step_size=0.001):
        """Simulates the denoising process."""
        pass  # NOT READY

    @staticmethod
    @abstractmethod
    def randn_lie_like(mtrx: Tensor, scale: Tensor | float = 1):
        """
        This method generates an appropraite random matrix in the Lie algebra,
        scales it by the specified factor, and computes the matrix exponential
        to obtain the corresponding Lie group element.

        Args:
            - mtrx (Tensor): The input tensor that defines the shape and
              device for the generated matrix.
            - scale (Tensor or float, optional): A scaling factor applied to
              the generated anti-Hermitian matrix. Default is 1.

        Returns:
            tuple: A tuple containing:
                - randn_grp (Tensor): The generated random Lie group element.
                - randn_alg (Tensor): The underlying Lie algebra element.
        """
        pass

    @property
    def denoising_flow(self):
        """This is a deterministic ODE, unlike denoising process."""
        return LieGroupDenoisingFlow(self.denoising_drift)


# =============================================================================
class UnitaryDiffusionProcess(LieGroupDiffusionProcess):
    """
    This class implements a diffusion process for generating unitary matrices
    based on the Lie group structure.
    """

    @staticmethod
    def randn_lie_like(mtrx: Tensor, scale: Tensor | float = 1):
        """
        This method generates a random anti-Hermitian matrix, scales it by
        the specified factor, and computes the matrix exponential to obtain
        the corresponding unitary matrix. The random anti-Hermitian matrix is
        generated using the `randn_antihermitian_like` function.

        Args:
            - mtrx (Tensor): The input tensor that defines the shape and
              device for the generated matrix.
            - scale (Tensor or float, optional): A scaling factor applied to
              the generated anti-Hermitian matrix. Default is 1.

        Returns:
            tuple: A tuple containing:
                - randn_grp (Tensor): The generated random unitary matrix.
                - randn_alg (Tensor): The underlying anti-Hermitian matrix.
        """
        randn_alg = scale * randn_antihermitian_like(mtrx)
        randn_grp = torch.matrix_exp(randn_alg)
        return randn_grp, randn_alg


class SUnitaryDiffusionProcess(LieGroupDiffusionProcess):
    """
    This class implements a diffusion process specifically for the special
    unitary group SU(n).
    """

    @staticmethod
    def randn_lie_like(mtrx: Tensor, scale: Tensor | float = 1):
        """
        This method generates a random traceless anti-Hermitian matrix, scales
        it by the specified factor, and computes the matrix exponential to
        obtain the corresponding SU(n) matrix. The random anti-Hermitian matrix
        is generated using the `randn_antihermitian_like` function.

        Args:
            - mtrx (Tensor): The input tensor that defines the shape and
              device for the generated matrix.
            - scale (Tensor or float, optional): A scaling factor applied to
              the generated anti-Hermitian matrix. Default is 1.

        Returns:
            tuple: A tuple containing:
                - randn_grp (Tensor): The generated random SU(n) matrix.
                - randn_alg (Tensor): The underlying traceless anti-Hermitian
                  matrix.
        """
        randn_alg = scale * make_traceless(randn_antihermitian_like(mtrx))
        randn_grp = torch.matrix_exp(randn_alg)
        return randn_grp, randn_alg


# =============================================================================
class LieGroupDenoisingFlow(LieGroupODEflow):
    """Denoising flow is a deterministic evolution of state variables.

    The super class has `forward` and `reverse` methods.
    """
    def __init__(self, denoising_drift, t_span=(1, 0), **kwargs):
        super().__init__(denoising_drift, t_span, method='Euler', **kwargs)


# =============================================================================
def randn_antihermitian_like(mtrx: Tensor) -> Tensor:
    """
    Generates random anti-Hermitian matrices with the same shape as the input.

    Both the real and imaginary parts of the non-diagonal elements are drawn
    from normal distributions with zero mean and variance of 1/2. The real part
    of the diagonal elements is zero, while the imaginary parts are drawn from
    a normal distribution with zero mean and unit variance. This normalization
    is consistent with generating an anti-Hermitian matrix using the generators
    of unitary matrices, where the coefficients are normally distributed with
    zero mean and unit variance, and the generators are normalized as
    :math:`Tr (T_a T_b) = - \delta_{a,b}`.

    By making the resulting matrix traceless, the variance of the imaginary
    parts of the diagonal elements is reduced, which is in agreement with the
    generation of a traceless anti-Hermitian matrix using the generators of
    special unitary matrices, where the coefficients are normally distributed
    with zero mean and unit variance.

    Args:
        mtrx (Tensor): Input tensor of shape (..., n, n) defining the desired
        shape and device of the output.

    Returns:
        Tensor: A random anti-Hermitian matrix of the same shape as the matrix.
    """
    assert mtrx.shape[-1] == mtrx.shape[-2], "Not a square matrix!"

    noise = torch.randn_like(mtrx) + 1j * torch.randn_like(mtrx)
    return (noise - noise.adjoint()) / 2


def randn_traceless_antihermitian_like(mtrx: Tensor) -> Tensor:
    """
    Generates random, traceless, anti-Hermitian matrices with the same shape
    as the input.

    For details see `randn_antihermitian_like` and `make_traceless`.
    """
    return make_traceless(randn_antihermitian_like(mtrx))


def make_traceless(mtrx: Tensor) -> Tensor:
    """
    Given a tensor of shape (..., n, n), makes the last two axes traceless
    by subtracting the mean of the diagonal elements from the diagonal entries.

    Args:
        mtrx (Tensor): Input tensor of shape (..., n, n).

    Returns:
        Tensor: A traceless tensor with the same shape as the matrix.
    """
    assert mtrx.shape[-1] == mtrx.shape[-2], "Not a square matrix!"

    # Compute the mean of diagonal elements -> reduced trace
    reduced_trace = mtrx.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    return mtrx - torch.diag_embed(reduced_trace.expand(mtrx.shape[:-1]))
