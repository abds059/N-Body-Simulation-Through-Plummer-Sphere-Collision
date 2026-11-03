from decomposition import decompose


def test_decompose():
    assert decompose(10, 3) == ([4, 3, 3], [0, 4, 7])
    for n, p in [(1000, 3), (1001, 8), (10000, 7), (8, 8)]:
        c, d = decompose(n, p)
        assert sum(c) == n and max(c) - min(c) <= 1
        assert all(d[i] + c[i] == d[i + 1] for i in range(p - 1))


if __name__ == "__main__":
    test_decompose()
    print("decomposition ok")