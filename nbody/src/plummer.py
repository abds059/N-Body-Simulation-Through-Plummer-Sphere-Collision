"""Plummer spheres and the two-cluster collision start. N-body units: G = 1."""
import numpy as np


def _unit_vectors(n, rng):
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - z * z)
    return np.column_stack((s * np.cos(phi), s * np.sin(phi), z))


def plummer(n, rng, mass=1.0, a=1.0):
    """n equal-mass bodies in a Plummer sphere (Aarseth, Henon & Wielen 1974), centred."""
    x = np.clip(rng.random(n), 1e-6, 0.999)  # mass fraction; outer 0.1% is cut off
    r = a / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    pos = r[:, None] * _unit_vectors(n, rng)
    q = np.empty(n)
    k = 0
    while k < n:  # rejection sampling of g(q) = q^2 (1 - q^2)^3.5
        c = rng.random(n)
        ok = c[rng.random(n) * 0.1 < c**2 * (1.0 - c**2) ** 3.5]
        take = min(len(ok), n - k)
        q[k:k + take] = ok[:take]
        k += take
    v_esc = np.sqrt(2.0 * mass) * (r**2 + a**2) ** -0.25
    vel = (q * v_esc)[:, None] * _unit_vectors(n, rng)
    return pos - pos.mean(axis=0), vel - vel.mean(axis=0), np.full(n, mass / n)


def two_clusters(n, seed=42, sep=8.0, vfrac=0.6, impact=1.0, a=1.0):
    """Two equal-mass Plummer spheres (total mass 1) on a bound collision course along x.

    sep    = distance between the cluster centres
    vfrac  = relative speed as a fraction of the pair's escape speed sqrt(2 G M / sep)
    impact = sideways offset along y
    """
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    p1, v1, m1 = plummer(n1, rng, 0.5, a)
    p2, v2, m2 = plummer(n2, rng, 0.5, a)
    v_rel = vfrac * np.sqrt(2.0 / sep)
    p1 += np.array([-sep / 2, -impact / 2, 0.0])
    p2 += np.array([sep / 2, impact / 2, 0.0])
    v1 += np.array([v_rel / 2, 0.0, 0.0])
    v2 += np.array([-v_rel / 2, 0.0, 0.0])
    pos = np.ascontiguousarray(np.vstack((p1, p2)))
    vel = np.ascontiguousarray(np.vstack((v1, v2)))
    mass = np.ascontiguousarray(np.concatenate((m1, m2)))
    label = np.concatenate((np.zeros(n1), np.ones(n2))).astype(np.int8)
    return pos, vel, mass, label