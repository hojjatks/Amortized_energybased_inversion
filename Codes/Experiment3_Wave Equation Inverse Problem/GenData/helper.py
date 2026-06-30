import numpy as np
import matplotlib.pyplot as plt

def observe_from_grid(P, time_idx, recv_idx):
    """
    P: (Nt, Nx)
    Returns:
      Y: (Nt_obs, Nr)
      y: (Nt_obs*Nr,)
      The vector y is ordered time-major: for each observation time (from early to late), the wavefield values at all receivers (from left to right) are concatenated before moving to the next time.
    """
    Y = P[np.ix_(time_idx, recv_idx)]   # (time, receiver)
    y = Y.reshape(-1)                   # time-major flatten
    return Y, y

import numpy as np

def get_Y(P, time_idx, recv_idx):
    """
    P: (Nt, Nx) wavefield
    time_idx: indices into time grid (length Nt_obs)
    recv_idx: indices into space grid (length Nr)

    Returns
    -------
    Y: (Nt_obs, Nr) where Y[k, j] = P[time_idx[k], recv_idx[j]]
    """
    return P[np.ix_(time_idx, recv_idx)]  # (time, receiver)

def plot_receiver_timeseries(t, Y, x_recv=None):
    """
    Plot one subplot per receiver: Y[:, j] vs t.

    t: (Nt_obs,) observation times
    Y: (Nt_obs, Nr)
    x_recv: optional (Nr,) receiver positions for titles
    """
    t = np.asarray(t)
    Y = np.asarray(Y)
    Nt_obs, Nr = Y.shape

    fig, axes = plt.subplots(nrows=Nr, ncols=1, figsize=(8, 1.8*Nr), sharex=True)
    if Nr == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(t, Y[:, j])
        if x_recv is None:
            ax.set_title(f"Receiver {j}")
        else:
            ax.set_title(f"Receiver {j} at x = {x_recv[j]:.4f}")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("p(t)")

    axes[-1].set_xlabel("time t")
    fig.tight_layout()
    return fig, axes



import numpy as np

def extract_arrival_times(Y, t, threshold=None, frac_peak=0.1):
    """
    Extract first-arrival times from wave traces.

    Parameters
    ----------
    Y : array, shape (N_t, N_receiver)
        Wavefield recorded at receivers over time.
        Y[:, i] is the trace at receiver i.

    t : array, shape (N_t,)
        Time vector corresponding to Y.

    threshold : float or None
        Absolute amplitude threshold for defining "arrival".
        If None, it is chosen automatically as a fraction of the peak.

    frac_peak : float
        If threshold is None, arrival is defined as the first time
        the signal exceeds frac_peak * max(|trace|).

    Returns
    -------
    T : array, shape (N_receiver,)
        Estimated arrival time at each receiver.
    """

    N_t, N_rec = Y.shape
    T = np.zeros(N_rec)

    for i in range(N_rec):

        trace = Y[:, i]

        # Choose threshold automatically if not provided
        if threshold is None:
            thresh = frac_peak * np.max(np.abs(trace))
        else:
            thresh = threshold

        # Find first time the absolute signal exceeds threshold
        idx = np.where(np.abs(trace) >= thresh)[0]

        if len(idx) == 0:
            # No arrival detected — set to NaN or last time
            T[i] = np.nan
        else:
            T[i] = t[idx[0]]

    return T

# ---- Example usage after you run the solver ----
# x, t, P = solve_wave_1d(...)
# recv_idx = ... (your interior equidistant indices)
# time_idx = ... (your observation time indices)

# Observation matrix
# Y = get_Y(P, time_idx, recv_idx)

# Corresponding time vector for those samples
# t_obs = t[time_idx]

# Receiver positions (optional, for labeling)
# x_recv = x[recv_idx]

# Plot
# fig, axes = plot_receiver_timeseries(t_obs, Y, x_recv=x_recv)
# plt.show()
# fig.savefig("./testfigs/receiver_timeseries.png", dpi=300)
def levelset_to_c_binary(U_sample, c_high, c_low):
    """
    Binary level-set mapping:
        if U > 0 -> c_high
        else     -> c_low
    """
    U_sample = np.asarray(U_sample)
    return np.where(U_sample > 0.0, c_high, c_low)