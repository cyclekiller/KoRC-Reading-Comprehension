# lcs_cython.pyx

def lcs(x, y):
    cdef int n, m, i, j
    cdef list f

    n, m = len(x), len(y)
    f = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n):
        for j in range(m):
            if x[i] == y[j]:
                f[i + 1][j + 1] = f[i][j] + 1
            else:
                f[i + 1][j + 1] = max(f[i + 1][j], f[i][j + 1])
    
    return f
