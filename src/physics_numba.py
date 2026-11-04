"""Compiled accel() with the same signature as physics.accel. Select with NBODY_KERNEL=numba."""
import numpy as np
from numba import njit


@njit(cache=True)
def accel(pos, mass, start, end, eps):
    n = pos.shape[0]
    acc = np.zeros((end - start, 3))
    eps2 = eps * eps
    for i in range(start, end):
        ax = ay = az = 0.0
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        for j in range(n):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx * dx + dy * dy + dz * dz + eps2
            f = mass[j] / (r2 * np.sqrt(r2))
            ax += f * dx
            ay += f * dy
            az += f * dz
        acc[i - start, 0] = ax
        acc[i - start, 1] = ay
        acc[i - start, 2] = az
    return acc