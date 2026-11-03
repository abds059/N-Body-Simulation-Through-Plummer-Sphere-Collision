"""Shared force and energy functions. G = 1. Used by sequential and MPI versions."""
import numpy as np


def accel(pos, mass, start, end, eps, block=256):
    """Acceleration on bodies [start:end) caused by ALL bodies (softened gravity)."""
    acc = np.empty((end - start, 3))
    eps2 = eps * eps
    for i in range(0, end - start, block):
        j = min(i + block, end - start)
        d = pos[None, :, :] - pos[start + i:start + j, None, :]  # (b, N, 3)
        r2 = np.einsum("bnk,bnk->bn", d, d) + eps2
        acc[i:j] = np.einsum("bn,bnk->bk", mass * r2**-1.5, d)
    return acc


def energy(pos, vel, mass, eps, block=256):
    """Returns (kinetic, potential), softened potential. Virial ratio = 2K / |W|."""
    kin = 0.5 * np.sum(mass * np.einsum("ij,ij->i", vel, vel))
    pot = 0.0
    for i in range(0, len(mass), block):
        d = pos[None, :, :] - pos[i:i + block, None, :]
        r = np.sqrt(np.einsum("bnk,bnk->bn", d, d) + eps * eps)
        pot -= np.sum(mass[i:i + block, None] * mass[None, :] / r)
    pot = 0.5 * (pot + np.sum(mass**2) / eps)  # drop the i == j terms, count each pair once
    return kin, pot

import os
if os.environ.get("NBODY_KERNEL") == "numba":
    from physics_numba import accel  # noqa: F811  same signature, chosen by environment variable