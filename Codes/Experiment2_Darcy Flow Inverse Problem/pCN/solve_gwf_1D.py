import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

#%%
def solve_gwf_1D(coef, F):
    coef = np.asarray(coef)
    F = np.asarray(F)
    K = len(coef)

    # ----------------------------------------------------------
    # 1. Cell-centered grid: x1(i) = (i - 0.5)/K
    # ----------------------------------------------------------
    x1 = (np.arange(1, K+1) - 0.5) / K

    # ----------------------------------------------------------
    # 2. Node-centered grid: x2(j) = (j - 1)/(K - 1)
    # ----------------------------------------------------------
    x2 = np.linspace(0.0, 1.0, K)

    # ----------------------------------------------------------
    # 3. Interpolate coef and F from cell centers → nodes
    #     MATLAB: interp1(x1, coef, x2, 'spline')
    # ----------------------------------------------------------
    coef2 = CubicSpline(x1, coef)(x2)
    F2    = CubicSpline(x1, F)(x2)

    # ----------------------------------------------------------
    # 4. Interior arrays
    # ----------------------------------------------------------
    N = K - 2
    coef_int = coef2[1:-1]
    F_int    = F2[1:-1]
    h = 1.0 / (K - 1)

    # ----------------------------------------------------------
    # 5. Build diffusion matrix A
    # ----------------------------------------------------------
    main_diag  = np.zeros(N)
    lower_diag = np.zeros(N-1)
    upper_diag = np.zeros(N-1)

    for i in range(N):
        a_ip = 0.5 * (coef2[i+2] + coef2[i+1])   # coef at (i+1/2)
        a_im = 0.5 * (coef2[i+1] + coef2[i])     # coef at (i-1/2)

        main_diag[i] = (a_ip + a_im) / h**2

        if i > 0:
            lower_diag[i-1] = -a_im / h**2
        if i < N-1:
            upper_diag[i] = -a_ip / h**2

    A = diags([main_diag, lower_diag, upper_diag],
              offsets=[0, -1, 1],
              shape=(N, N),
              format='csc')

    # ----------------------------------------------------------
    # 6. Solve system
    # ----------------------------------------------------------
    Pint = spsolve(A, F_int)

    # ----------------------------------------------------------
    # 7. Apply boundary conditions: p(0)=p(1)=0
    # ----------------------------------------------------------
    P2 = np.zeros(K)
    P2[1:-1] = Pint

    # ----------------------------------------------------------
    # 8. Interpolate solution back to cell centers
    #     MATLAB: interp1(x2, P2, x1, 'spline')
    # ----------------------------------------------------------
    P = CubicSpline(x2, P2)(x1)

    return P
