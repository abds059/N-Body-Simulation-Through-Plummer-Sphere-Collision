"""Block decomposition: counts and displacements (in bodies) for n bodies on p ranks."""


def decompose(n, p):
    base, extra = divmod(n, p)  # the first n % p ranks get one extra body
    counts = [base + (r < extra) for r in range(p)]
    displs = [sum(counts[:r]) for r in range(p)]
    return counts, displs