import numpy as np
from scipy.fft import idct

def gaussrnd(alpha: float, tau: float, N: int, scale = 1) -> np.ndarray:
    """
    Return a sample of a Gaussian random field on [0,1] with:
        mean 0
        covariance operator C = (-Delta + tau^2)^(-alpha)
    where Delta is the Laplacian with zero Neumann boundary conditions.
    """
    # Random variables in KL expansion
    xi = np.random.normal(loc=0.0, scale=1.0, size=N)

    # Define the (square root of) eigenvalues of the covariance operator
    K1 = np.arange(N)
    coef = (np.pi**2 * (K1**2) + tau**2)**(-alpha / 2.0)

    # Construct the KL coefficients
    L = np.sqrt(N) * coef * xi * scale
    L[0] = 0.0

    # Inverse DCT (Type-II, matching MATLAB's 'Type',2)
    U = idct(L, type=2, norm='ortho')  # you can add norm='ortho' if you want orthonormal scaling

    return U
