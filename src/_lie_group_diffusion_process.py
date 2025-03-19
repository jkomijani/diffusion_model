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

    where :math:`y(t)` is the state variable as a Lie-group member,
    :math:`\sigma = \sqrt{2\omega}`, and :math:`\eta_t` represents noise.
    This diffusion process governs the evolution of a state on a Lie group,
    with the time-dependent, variance-expanding noise.
    """

    def __init__(
        self,
        score_func: torch.nn.Module,
        omega: float = 1.0,
        sigma: float = 2**0.5
    ):
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
        self.sigma = sigma

    def denoising_drift(self, t: Tensor, y_t: Tensor, sigma_tilde: float = 0):
        """
        Computes the drift term in the denoising process.

        Args:
            - t (Tensor): A 1-dimensional tensor representing the time,
              corresponding to the batch axis of `y_t`.
            - y_t (Tensor): The current state of the system at time `t`.
            - sigma_tilde (float, optional): A scaling factor to adjust the
              noise. Default is `0`, meaning no noise.

        Returns:
            Tensor: The drift term, which is used in the denoising process.
        """
        coeff = (self.sigma**2 + sigma_tilde**2) / 2
        scaled_coeff = coeff * torch.exp(2 * self.omega * t)

        return - scaled_coeff * self.score_func(t, y_t)

    def run_for_training(self, y_0: Tensor, t_steps: Tensor, step_size: float):
        r"""
        Simulates the forward diffusion process for training purposes.

        This method simulates the forward diffusion process starting from an
        initial state `y_0`, which is a tensor of the Lie group elements, over
        discrete time steps `t_steps`. The method computes noisy states and
        their noise characteristics, which can be used for training denoising
        models.

        Args:
            - y_0 (Tensor): The initial state of the system at time `t = 0`.
            - t_steps (Tensor): A 1D tensor of random time steps corresponding
              to the batch axis of `y_0`.
            - step_size (float): The step size used for numerical integration.

        Returns:
            tuple:
                - Tensor: `y_t` (noisy state): The noisy states after applying
                  the diffusion process over the specified time steps.

                - Tensor: `alg / std` (normalized algebraic state): The state
                  in the tangent space of the Lie group, normalized by its
                  standard deviation. This tensor shows how noise evolves in
                  the algebraic space over time.
                  We have :math:`alg_t = \int_0^t d \Gamma_t`.

                - Tensor: `std`: The accumulated standard deviation (noise)
                  over time. This tensor tracks how much noise has been added
                  to the algebraic state during the diffusion process.
        """
        assert t_steps.shape[0] == y_0.shape[0]

        max_t_steps = torch.max(t_steps)
        t_steps = t_steps.view(-1, *[1] * (y_0.ndim - 1))

        # Calculate std of the integrated noise of the frist step
        w = self.omega  # Diffusion rate parameter
        std_1 = self.sigma * np.sqrt((np.exp(2 * w * step_size) - 1) / (2 * w))

        # Initialize time, algebraic state, and variance
        t_0 = 0  # Initial time set to 0
        alg = torch.zeros_like(y_0)  # the algebra-valued state variable

        # Loop over the maximum number of steps
        for step in range(1, 1 + max_t_steps):

            # Calculate std of the integrated noise from `t_0` to `t_0 + dt`
            # Then, suppress std if time is larger than the evaluation time
            std_step = std_1 * np.exp(w * t_0)
            std = std_step * (step <= t_steps)

            # Generate noise with the specified std
            randn_grp, randn_alg = self.randn_lie_like(y_0.real, scale=std)

            # Update the group-valued & algebra-valued states and time
            y_0 = randn_grp @ y_0
            alg = randn_alg + alg
            t_0 += step_size

        t_eval = step_size * t_steps
        std = self.sigma * np.sqrt((np.exp(2 * w * t_eval) - 1) / (2 * w))

        return y_0, alg / std, std

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
        # Calculate std of the integrated noise of the frist step
        w = self.omega  # Diffusion rate parameter
        std_1 = self.sigma * np.sqrt((np.exp(2 * w * step_size) - 1) / (2 * w))

        y_eval = [None] * len(t_eval)  # Initialize list for evaluated states

        # Iterate through evaluation times
        for ind, t in enumerate(t_eval):

            assert t >= t_0, "`t_eval` must monotonically increase."

            while t > t_0:  # Iterate until reaching the target time

                # Calculate std of integrated noise from `t_0` to `t_0 + dt`
                std = std_1 * np.exp(w * t_0)

                # Generate noise with the specified std
                randn_grp, _ = self.randn_lie_like(y_0.real, scale=std)

                # Update the group-valued state and time
                y_0 = randn_grp @ y_0
                t_0 += step_size

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

    def denoising_flow(self, t_span=(1, 0), method='Euler', step_size=0.001):
        """
        Creates and returns an instance of `LieGroupODEflow` that performs
        denoising using a deterministic ordinary differential equation (ODE).

        Unlike `run_denoising_process`, which solves a stochastic differential
        equation (SDE) for denoising, this method follows a deterministic
        approach.

        Note that the `ODEflow` class has `forward` and `reverse` methods.
        """
        kwargs = {'method': method, 'step_size': step_size}
        return LieGroupODEflow(self.denoising_drift, t_span, **kwargs)


# =============================================================================
class UnitaryDiffusionProcess(LieGroupDiffusionProcess):
    """
    This class implements a diffusion process for generating unitary matrices
    based on the Lie group structure.

    NOTE: Instead of using this class, we recommend separting the group to
    `U(1) x SU(n)` and evaluating the process for each sub-group separately.
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


class SUnDiffusionProcess(LieGroupDiffusionProcess):
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
def randn_antihermitian_like(mtrx: Tensor) -> Tensor:
    r"""
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
