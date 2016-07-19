import numpy as np
import scipy.linalg   as spla
from scipy.spatial.distance import cdist

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[ 0 ]))

def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))

def compute_kernel(lls, lsf, x, z):

    ls = np.exp(lls)
    sf = np.exp(lsf)

    if x.ndim == 1:
        x= x[ None, : ]

    if z.ndim == 1:
        z= z[ None, : ]

    r2 = cdist(x, z, 'seuclidean', V = ls)**2.0  
    k = sf * np.exp(-0.5*r2)
    return k

def compute_psi1(lls, lsf, xmean, xvar, z):

    if xmean.ndim == 1:
        xmean = xmean[ None, : ]

    ls = np.exp(lls)
    sf = np.exp(lsf)
    lspxvar = ls + xvar
    constterm1 = ls / lspxvar
    constterm2 = np.prod(np.sqrt(constterm1))
    r2_psi1 = cdist(xmean, z, 'seuclidean', V = lspxvar)**2.0  
    psi1 = sf*constterm2*np.exp(-0.5*r2_psi1)
    return psi1

def compute_psi2(lls, lsf, xmean, xvar, z):

    ls = np.exp(lls)
    sf = np.exp(lsf)
    lsp2xvar = ls + 2.0 * xvar
    constterm1 = ls / lsp2xvar
    constterm2 = np.prod(np.sqrt(constterm1))

    n_psi = z.shape[ 0 ]
    v_ones_n_psi = np.ones(n_psi)
    v_ones_dim = np.ones(z.shape[ 1 ])

    D = ls
    Dnew = ls / 2.0
    Btilde = 1.0 / (Dnew + xvar)
    Vtilde = Btilde - 1.0 / Dnew
    Qtilde = 1.0 / D + 0.25 * Vtilde

    T1 = -0.5 * np.outer(np.dot((z**2) * np.outer(v_ones_n_psi, Qtilde), v_ones_dim), v_ones_n_psi)
    T2 = +0.5 * np.outer(np.dot(z, xmean * Btilde), v_ones_n_psi)
    T3 = -0.25 * np.dot(z * np.outer(v_ones_n_psi, Vtilde), z.T)
    T4 = -0.5 * np.sum((xmean**2) * Btilde) 

    M = T1 + T1.T + T2 + T2.T + T3 + T4

    psi2 = sf**2.0 * constterm2 * np.exp(M)
    return psi2

def d_trace_MKzz_dhypers(lls, lsf, z, M, Kzz):

    dKzz_dlsf = Kzz
    ls = np.exp(lls)

    # This is extracted from the R-code of Scalable EP for GP Classification by DHL and JMHL

    gr_lsf = np.sum(M * dKzz_dlsf)

    # This uses the vact that the distance is v^21^T - vv^T + 1v^2^T, where v is a vector with the l-dimension
    # of the inducing points. 

    Ml = 0.5 * M * Kzz
    Xl = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / np.sqrt(ls))
    gr_lls = np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml.T, Xl**2)) + np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml, Xl**2)) \
    - 2.0 * np.dot(np.ones(Xl.shape[ 0 ]), (Xl * np.dot(Ml, Xl)))

    Xbar = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / ls)
    Mbar1 = - M.T * Kzz
    Mbar2 = - M * Kzz
    gr_z = (Xbar * np.outer(np.dot(np.ones(Mbar1.shape[ 0 ]) , Mbar1), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar1, Xbar)) +\
        (Xbar * np.outer(np.dot(np.ones(Mbar2.shape[ 0 ]) , Mbar2), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar2, Xbar))

    # The cost of this function is dominated by five matrix multiplications with cost M^2 * D each where D is 
    # the dimensionality of the data!!!

    return gr_lsf, gr_lls, gr_z