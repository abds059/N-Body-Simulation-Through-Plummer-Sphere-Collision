import numpy as np

from physics import accel, energy
from plummer import plummer, two_clusters


def test_two_body():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([0.5, 0.5])
    a = accel(pos, mass, 0, 2, eps=1e-6)
    assert np.allclose(a, [[0.5, 0, 0], [-0.5, 0, 0]], rtol=1e-9)  # G m / d^2 = 0.5


def test_slices_and_blocks_match_full():
    pos, vel, mass, _ = two_clusters(60)
    full = accel(pos, mass, 0, 60, 0.05)
    parts = np.vstack([accel(pos, mass, 0, 17, 0.05), accel(pos, mass, 17, 60, 0.05)])
    small_blocks = accel(pos, mass, 0, 60, 0.05, block=7)
    assert np.allclose(full, parts, rtol=0, atol=1e-13)
    assert np.allclose(full, small_blocks, rtol=0, atol=1e-13)


def test_virial_ratio():
    pos, vel, mass = plummer(2000, np.random.default_rng(1))
    k, w = energy(pos, vel, mass, 0.05)
    q = 2 * k / abs(w)
    print("virial ratio 2K/|W| =", round(q, 3))
    assert 0.8 < q < 1.2


if __name__ == "__main__":
    test_two_body()
    test_slices_and_blocks_match_full()
    test_virial_ratio()
    print("all tests passed")