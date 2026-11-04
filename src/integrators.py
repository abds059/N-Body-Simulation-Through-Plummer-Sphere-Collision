"""Leapfrog (kick-drift-kick) pieces, in place. MPI ranks reuse these on their own slice."""


def kick(vel, acc, half_dt):
    vel += half_dt * acc


def drift(pos, vel, dt):
    pos += dt * vel