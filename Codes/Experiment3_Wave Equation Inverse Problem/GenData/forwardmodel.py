import numpy as np
import matplotlib.pyplot as plt
from gaussrnd import *
# -----------------------------
# Ricker wavelet
# -----------------------------
def ricker(t, f0):
    """
    Ricker wavelet evaluated at time(s) t with dominant frequency f0.
    t can be a scalar or a NumPy array.
    """
    t = np.asarray(t)
    t0 = 3.0 / f0
    arg = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


# -----------------------------
# Laplacian matrix with Neumann BC
# -----------------------------
def build_L_neumann_matrix(Nx, dx):
    """
    Build the 1D Laplacian matrix L (Nx x Nx) with homogeneous Neumann BC:

      p_x(0) = 0, p_x(L) = 0

    Discretization:
      - interior i = 1..Nx-2: (p_{i+1} - 2 p_i + p_{i-1}) / dx^2
      - left boundary i = 0 : 2 (p_1   - p_0    ) / dx^2
      - right boundary i = N-1: 2 (p_{N-2} - p_{N-1}) / dx^2
    """
    L = np.zeros((Nx, Nx))

    # interior rows
    for i in range(1, Nx - 1):
        L[i, i - 1] = 1.0
        L[i, i]     = -2.0
        L[i, i + 1] = 1.0

    # left boundary: p_xx(0) ≈ 2 (p_1 - p_0) / dx^2  => [-2, 2, 0, ...]
    L[0, 0] = -2.0
    L[0, 1] =  2.0

    # right boundary: p_xx(L) ≈ 2 (p_{N-2} - p_{N-1}) / dx^2 => [..., 2, -2]
    L[-1, -2] =  2.0
    L[-1, -1] = -2.0

    return L / dx**2


# -----------------------------
# Wave solver (takes u and Lmat)
# -----------------------------
def solve_wave_1d(
    c,
    Lmat,
    L=1.0,
    T=1.0,
    f0=20.0,
    xs_frac=0.25,
    dt=0.001
):
    """
    Solve: p_tt = u(x)^2 p_xx + delta(x-xs)*Ricker(t)
    on x in [0, L], t in [0, T], with

      - homogeneous Neumann BC encoded in Lmat
      - zero initial displacement and velocity
      - leapfrog time stepping

    Parameters
    ----------
    c : (Nx,) array
        Wave speed values on the spatial grid.
    Lmat : (Nx, Nx) array
        Laplacian matrix with Neumann BC (from build_L_neumann_matrix).
    L : float
        Domain length.
    T : float
        Final time.
    f0 : float
        Dominant frequency of the Ricker source.
    xs_frac : float
        Source location as fraction of L (e.g. 0.25 -> x_s = L/4).

    Returns
    -------
    x : (Nx,) array
        Spatial grid.
    t : (Nt,) array
        Time grid.
    P : (Nt, Nx) array
        Wavefield, P[n, i] = p(x[i], t[n]).
    """
    c = np.asarray(c)
    Nx = c.size

    # spatial grid (must be consistent with dx used in Lmat)
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]

    c2 = c**2
    Nt = int(T / dt)
    t = np.arange(Nt) * dt
    cmax = np.max(np.abs(c))
    cfl = cmax * dt / dx
    if cfl > 1.0:
        raise ValueError(f"CFL too large: cmax*dt/dx = {cfl:.3f} > 1.0. Reduce dt.")

    # initial conditions: p = 0, p_t = 0
    p_prev = np.zeros(Nx)  # p at t^{n-1}
    p_curr = np.zeros(Nx)  # p at t^{n}
    P = np.zeros((Nt, Nx))

    # source index
    xs = xs_frac * L
    isrc = np.argmin(np.abs(x - xs))

    # precompute Ricker wavelet at all times
    rick_t = ricker(t, f0)

    for n in range(Nt):
        # Laplacian via matrix-vector multiply
        lap = Lmat @ p_curr

        # source term (point source at isrc)
        src = np.zeros_like(p_curr)
        src[isrc] = rick_t[n]

        # leapfrog update
        if n == 0:
            # first step via Taylor expansion
            p_next = p_curr + 0.5 * dt**2 * (c2 * lap + src)
        else:
            p_next = 2.0 * p_curr - p_prev + dt**2 * (c2 * lap + src)

        # store current field
        P[n, :] = p_curr

        # shift in time
        p_prev, p_curr = p_curr, p_next

    return x, t, P


# L = 1.0
# Nx = 201
# dx = L / (Nx - 1)
# scale=20
# # constant wave speed u(x) = 2
# x = np.linspace(0.0, L, Nx)
# c = np.exp(gaussrnd(2,5,Nx,scale))
# print(c)
# # build Laplacian matrix once
# Lmat = build_L_neumann_matrix(Nx, dx)

# # run solver
# x, t, P = solve_wave_1d(
#     c=c,
#     Lmat=Lmat,
#     L=L,
#     T=1.0,
#     f0=20.0,
#     xs_frac=0.25,
#     dt=0.001
# )

# plt.figure()
# plt.plot(c)
# plt.savefig("./testfigs/c.png")
# print("x shape:", x.shape)
# print("t shape:", t.shape)
# print("t0 ", t[0])
# print("P shape:", P.shape)
# plt.figure()
# plt.imshow(P)
# plt.savefig("./testfigs/spatiotemporal",dpi=500)