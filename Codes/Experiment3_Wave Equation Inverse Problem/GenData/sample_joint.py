import numpy as np 
from gaussrnd import *
import matplotlib.pyplot as plt
from forwardmodel import *
import time
from helper import *
np.random.seed(1)

# specify the prior:
alpha=2
tau=5 
scale=10

# level sets for wave speed: see develope_sample_joint.ipynb to understand where these number come from
c_high = np.exp(0.27)
c_low  = np.exp(-0.27)


# solver adjustment:

N = 100 # number of spatial intervals
dt = 2e-4
L=1
f0=20
dx=L/N
xs_frac=0.5
T=1
Np1= N+1 # number of grid points including the boundary
i = np.arange(1, N + 1)
x_center = (i - 0.5) / N  # shape (N,) x at the center points
i = np.arange(0, Np1 )
x_nodes = i*dx
Lmat = build_L_neumann_matrix(Np1, dx) # solver setting (call once)
# observation:
# --- spatial receivers (indices on x grid) ---
Nr = 10
Nrp2 = Nr+2
i = np.arange(0 , Nrp2)
dx_obs=L/(Nr+1)
ratio_obs_to_discritization= int(dx_obs/dx)
x_nodes_rec = i*dx_obs

recv_idx = (i*ratio_obs_to_discritization)[1:-1]
x_recv = x_nodes[recv_idx]
print(recv_idx)
# --- time samples (indices on t grid) ---
dt_obs = dt                     
time_idx = np.arange(0, int(T/dt_obs)) * int(dt_obs/dt)
t_recv = time_idx * dt


Ny = len(time_idx) * len(recv_idx)


# sampling:
Nsample=2_000_000 
U_samples_center = np.zeros((N, Nsample))
Arrival_store=np.zeros((Nr,Nsample))
for i in range(Nsample):
    U_samples_center[:, i] = gaussrnd(alpha, tau, N, scale=scale)
    # interpolate to get u at the nodes
    U_sample = np.interp(x_nodes, x_center, U_samples_center[:, i])
    c = levelset_to_c_binary(U_sample,c_high=c_high,c_low=c_low)
    # solve the PDE
    # t1=time.time()
    x, t, P = solve_wave_1d(c=c, Lmat=Lmat, L=L, T=T, f0=f0, xs_frac=xs_frac, dt=dt)
    # t2=time.time()
    # print(t2-t1)
    Y,y=observe_from_grid(P, time_idx, recv_idx) # I dont want to add noise to the data anymore, I will add it later, during the training.
    # y = y + scale_obs_noise * np.random.randn(y.size)
    ArrivalTimes=extract_arrival_times(Y, t_recv) # extracting the arrival times. 
    Arrival_store[:, i] = ArrivalTimes


np.savez(
    "./../../../Data/Experiment7/samples_wave_center.npz",
    U=U_samples_center,                # (Np1, Nsample) : latent field samples at nodes
    arrival=Arrival_store,
    x_nodes=x_nodes,          # solver spatial grid
    t_grid=t,                 # solver time grid (from last run; same each run)
    recv_idx=recv_idx,        # receiver indices on x_nodes
    x_recv=x_recv,            # receiver positions
    time_idx=time_idx,        # time indices on t_grid
    t_recv=t_recv,            # observation times
    alpha=alpha,
    tau=tau,
    scale=scale,
    dt=dt,
    T=T,
    L=L,
    f0=f0,
    xs_frac=xs_frac,
    dt_obs=dt_obs)
print("Saved to joint_samples_wave_center.npz")
