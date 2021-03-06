# this is a faster version of the Cahn-Hilliard math
# using numba

import numpy as np
from numba import jit

@jit
def defF(F, dx):
    # get the dimensions of the image
    nx = F.shape[0]
    ny = F.shape[1]
    dF = np.zeros(shape=F.shape, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            lj = j % ny
            lVal = -4*F[i,j]
            # handle PBCs with modulo
            li = (i + 1) % nx
            lVal += F[li, j]
            li = (i - 1) % nx
            lVal += F[li, j]
            lj = (j + 1) % ny
            lVal += F[i, lj]
            lj = (j - 1) % ny
            lVal += F[i, lj]
            dF[i] = lVal / (dx*dx)
    return dF
