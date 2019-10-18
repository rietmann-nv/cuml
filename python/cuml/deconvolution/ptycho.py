import jax.numpy as np
from jax import random

def compute_psi(Pj, O):
    (np, mp) = Pj.shape
    (no, mo) = O.shape

    # number of overlapping multiplications
    ni = no - np + 1
    mi = mo - mp + 1

    psi = np.zeros((ni*np, mi*mp), dtype=np.complex64)
    
    # each "probe" 'Pj' is multiplied against 'O' pointwise. Accumulate the tiled result in a list.
    for i_n in range(ni):
        psi.append([])
        for i_m in range(mi):
            psi_nm = Pj * O[i_n:i_n+np, i_m:i_m+mp]
            psi[i_n].append(psi_nm)

    return psi

def psi_transpose(psi):
    """psi is computed over i, then over j. However M, is computed over j, then I.
    This function flips order of computed psi.
    """

    nj = len(psi)
    ni = len(psi[0])
    mi = len(psi[0][0])

    # init list
    psiT = [None] * ni
    for i_n in range(ni):
        psiT[i_n] = [None] * mi

    for i_n in range(ni):
        for i_m in range(mi):
            psiT[i_n][i_m] = []
            for j in range(nj):
                psiT[i_n][i_m].append(psi[j][i_n][i_m])

    return psiT
    

def compute_Mi(psi, nj, O_shape):

    Mi2 = np.zeros(O_shape, dtype=np.complex64)
    for j in range(nj):
        Mi2 += np.abs(np.fft.fftn(psi[j]))**2

    Mi = np.sqrt(Mi2)

    return Mi
