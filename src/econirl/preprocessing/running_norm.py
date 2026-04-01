"""Running normalization for online feature standardization.

Implements numerically stable running mean and variance using the
parallel algorithm from Chan, Golub, and LeVeque (1979). This avoids
catastrophic cancellation that arises from naive one-pass formulas
when the mean is large relative to the variance.

The imitation library (HumanCompatibleAI/imitation) uses this pattern
for reward and observation normalization during training. In econirl
it is useful for normalizing feature matrices before IRL estimation,
where feature scales can vary by orders of magnitude across columns.

Reference:
    Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). "Updating
    formulae and a pairwise algorithm for computing sample variances."
    Technical Report STAN-CS-79-773, Stanford University.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class RunningNorm:
    """Online mean/variance tracker with numerically stable updates.

    Tracks running statistics and can normalize arrays to zero mean
    and unit variance. Works with both JAX and NumPy arrays.

    Parameters
    ----------
    size : int
        Dimensionality of the input (number of features).
    eps : float
        Small constant added to standard deviation to prevent
        division by zero. Default 1e-8.

    Example
    -------
    >>> norm = RunningNorm(size=3)
    >>> norm.update(features_batch_1)
    >>> norm.update(features_batch_2)
    >>> normalized = norm.normalize(features_batch_3)
    """

    def __init__(self, size: int, eps: float = 1e-8):
        self._size = size
        self._eps = eps
        self._count = 0
        self._mean = np.zeros(size, dtype=np.float64)
        self._M2 = np.zeros(size, dtype=np.float64)

    @property
    def mean(self) -> np.ndarray:
        """Current running mean, shape (size,)."""
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Current running variance, shape (size,)."""
        if self._count < 2:
            return np.ones(self._size, dtype=np.float64)
        return self._M2 / self._count

    @property
    def std(self) -> np.ndarray:
        """Current running standard deviation, shape (size,)."""
        return np.sqrt(self.var)

    @property
    def count(self) -> int:
        """Number of samples seen so far."""
        return self._count

    def update(self, x: np.ndarray | jnp.ndarray) -> None:
        """Update running statistics with a new batch of observations.

        Uses the parallel algorithm (Chan et al. 1979) which combines
        two sets of statistics without revisiting individual samples.

        Parameters
        ----------
        x : array-like
            New observations, shape (batch_size, size) or (size,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(axis=0)

        # Chan et al. parallel combination formula
        delta = batch_mean - self._mean
        new_count = self._count + batch_count

        if new_count == 0:
            return

        self._mean = self._mean + delta * (batch_count / new_count)
        self._M2 = (
            self._M2
            + batch_M2
            + delta ** 2 * (self._count * batch_count / new_count)
        )
        self._count = new_count

    def normalize(self, x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """Normalize an array to zero mean, unit variance.

        Parameters
        ----------
        x : array-like
            Input array, shape (..., size).

        Returns
        -------
        jnp.ndarray
            Normalized array with same shape as input.
        """
        mean = jnp.array(self._mean)
        std = jnp.array(self.std + self._eps)
        return (jnp.asarray(x) - mean) / std

    def denormalize(self, x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """Reverse normalization to recover original scale.

        Parameters
        ----------
        x : array-like
            Normalized array, shape (..., size).

        Returns
        -------
        jnp.ndarray
            Denormalized array with same shape as input.
        """
        mean = jnp.array(self._mean)
        std = jnp.array(self.std + self._eps)
        return jnp.asarray(x) * std + mean

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self._count = 0
        self._mean = np.zeros(self._size, dtype=np.float64)
        self._M2 = np.zeros(self._size, dtype=np.float64)
