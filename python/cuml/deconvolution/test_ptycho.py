from ptycho import compute_psi, psi_transpose, compute_Mi

import jax.numpy as np
from jax import random

def test_psi():
    key = random.PRNGKey(0)

    O = np.full((9,9), 1.0, dtype=np.complex64)
    Nj = 3
    P1 = np.full((3,3), 41.0, dtype=np.complex64)
    P2 = np.full((3,3), 42.0, dtype=np.complex64)
    P3 = np.full((3,3), 43.0, dtype=np.complex64)

    P=[P1, P2, P3]
    psi = []
    for Pj in P:
        psi_j = compute_psi(Pj, O)
        psi.append(psi_j)

    print("psi=", psi)

    psiT = psi_transpose(psi)

    M = 9*[None]
    for i_n in range(9):
        M[i_n] = 9*[None]

    for i_n in range(9):
        for i_m in range(9):
            M[i_n][i_m] = compute_Mi(psiT[i_n][i_m], 3, O.shape)

    print M
