#!/usr/bin/env python3
# ============================================================
# scaling_models_params_two_losses.py
#
# Runs a PARAMETER-scaling sweep for TWO losses:
#   1) "joint"  : energy_distance_loss(model(u,y,y_dag))
#   2) "prior"  : energy_distance_loss_fromprior(model(u,y_dag))
#
# Here:
#   - depth is fixed at 2
#   - width is varied
#   - data sizes are fixed to K_TRAINS = [500, 8000, 64000]
#
# Produces TWO plots (VAL and TEST), each overlaying:
#   - solid lines  : joint loss
#   - dashed lines : prior loss
# for each training data size K.
#
# x-axis = number of parameters
# y-axis = avg energy distance metric
#
# Everything saved under:
#   ./param_scaling_saved_<RUN_TAG>/
#       meta.json
#       scaling_models_params_two_losses.py (copy)
#       results_models_param_scaling_two_losses.json
#       y_kappa_val.npy
#       y_kappa_test.npy
#       checkpoints/
#           joint/
#           prior/
#       plots/
# ============================================================

import os, json, time, shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -------------------------
# Config (edit these)
# -------------------------
DATA_DIR   = "./../../../Data/Experiment1"
U_PATH     = os.path.join(DATA_DIR, "samples_u.npy")
Y_PATH     = os.path.join(DATA_DIR, "samples_y.npy")

# PARAMETER scaling configs: FIX DEPTH=2, VARY WIDTH
MODEL_CONFIGS = [
    dict(hidden_dim=8,   num_layers=2),
    dict(hidden_dim=16,  num_layers=2),
    dict(hidden_dim=32,  num_layers=2),
    dict(hidden_dim=64,  num_layers=2),
    dict(hidden_dim=128, num_layers=2),
]

# MODEL_CONFIGS = [
#     dict(hidden_dim=8,   num_layers=2),
#     dict(hidden_dim=16,  num_layers=2),
# ]

# Only three training sizes
K_TRAINS = [500, 8000, 64000]
SEEDS    = [0, 1, 2]
# K_TRAINS = [500, 8000]
# SEEDS    = [0, 1]
# Two losses to compare
LOSS_MODES = ["joint", "prior"]
LINESTYLE  = {"joint": "-", "prior": "--"}
LOSS_LABEL = {"joint": "joint", "prior": "prior"}

# Training
BATCH_SIZE = 700
LR         = 1e-3
MAX_EPOCHS = 50
PATIENCE   = 50

# Fixed split sizes
VAL_SIZE  = 1000
TEST_SIZE = 1000

# Evaluation metric
NY_KAPPA   = 30
NP_PUSH    = 5000
PUSH_BATCH = 1024
U_SPAN     = 20.0
N_GRID     = 4001

# Inverse problem params (Experiment 1)
M0     = 0.0
SIGMA0 = 1.0
SIGMA  = 1.0
G = lambda u: u**2

# -------------------------
# Device + output folders
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"./param_scaling_saved_{RUN_TAG}"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

for lm in LOSS_MODES:
    os.makedirs(os.path.join(CKPT_DIR, lm), exist_ok=True)

print("Using device:", device)
print("OUT_DIR:", os.path.abspath(OUT_DIR))

# save script into OUT_DIR
try:
    shutil.copy(__file__, os.path.join(OUT_DIR, "scaling_models_params_two_losses.py"))
except Exception as e:
    print("Warning: could not copy __file__. Error:", e)

# -------------------------
# Dataset
# -------------------------
class IndexedJointDataset(torch.utils.data.Dataset):
    def __init__(self, u, y):
        self.u = torch.tensor(u, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.y[idx], idx

# -------------------------
# Model
# -------------------------
class MLP_T(nn.Module):
    """
    mode="joint": forward(u, y, y_dag) uses input dim 3: [u, y, y_dag]
    mode="prior": forward(u, y_dag)    uses input dim 2: [u, y_dag]
                 (implemented as forward(u, y, y_dag=None) with y==y_dag)
    """
    def __init__(self, hidden_dim=64, num_layers=4, mode="joint"):
        super().__init__()
        assert mode in ("joint", "prior")
        self.mode = mode
        in_dim = 3 if mode == "joint" else 2

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u, y, y_dag=None):
        if self.mode == "joint":
            if y_dag is None:
                raise ValueError("mode='joint' requires y_dag")
            x = torch.stack([u, y, y_dag], dim=-1)
        else:
            # here y is interpreted as y_dag
            x = torch.stack([u, y], dim=-1)
        return u + self.net(x).squeeze(-1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# OOM-safe repulsion helper
# -------------------------
def mean_abs_pairwise_torch(v):
    """
    v: (B,) tensor
    returns scalar = mean_{i,j} |v_i - v_j|
    Uses sorting trick O(B log B), O(B) memory.
    """
    v_sorted, _ = torch.sort(v.view(-1))
    B = v_sorted.numel()
    k = torch.arange(1, B + 1, device=v.device, dtype=v.dtype)
    pair_sum = torch.sum(v_sorted * (2*k - B - 1))
    return (2.0 * pair_sum) / (B * B)

# ============================================================
# TRAINING LOSSES
# ============================================================
def energy_distance_loss(model, u_batch, y_batch):
    """
    JOINT loss:
      - Accuracy excludes diagonal
      - Repulsion averages over k of mean_{i,j} |T_{i,k} - T_{j,k}|
    """
    device = u_batch.device
    B = u_batch.shape[0]

    # Accuracy term over (i,j), excluding i=j
    u_i = u_batch.unsqueeze(1).expand(B, B).reshape(-1).to(device)
    y_i = y_batch.unsqueeze(1).expand(B, B).reshape(-1).to(device)
    y_j = y_batch.unsqueeze(0).expand(B, B).reshape(-1).to(device)
    u_j = u_batch.unsqueeze(0).expand(B, B).reshape(-1).to(device)

    acc_pred = model(u_i, y_i, y_j)
    acc_matrix = torch.abs(acc_pred - u_j).view(B, B)
    acc_matrix.fill_diagonal_(0.0)
    acc_loss = acc_matrix.sum() / (B * (B - 1))

    # Repulsion term
    y_k = y_batch.view(1, B).expand(B, B)
    u_i_mat = u_batch.view(B, 1).expand(B, B)
    y_i_mat = y_batch.view(B, 1).expand(B, B)

    T_out = model(
        u_i_mat.reshape(-1),
        y_i_mat.reshape(-1),
        y_k.reshape(-1),
    ).view(B, B)

    rep_k = []
    for k in range(B):
        rep_k.append(mean_abs_pairwise_torch(T_out[:, k]))
    repulsion = torch.stack(rep_k).mean() * (B / (B - 1))

    return 2.0 * acc_loss - repulsion

def energy_distance_loss_fromprior(model, u_batch, y_batch):
    """
    PRIOR loss:
      - model called as model(u, y_dag)
      - Accuracy excludes diagonal
      - Repulsion computed per column with sorting trick
    """
    device = u_batch.device
    B = u_batch.shape[0]

    # Accuracy term over (i,j), excluding i=j
    u_i = u_batch.unsqueeze(1).expand(B, B).reshape(-1).to(device)
    y_j = y_batch.unsqueeze(0).expand(B, B).reshape(-1).to(device)
    u_j = u_batch.unsqueeze(0).expand(B, B).reshape(-1).to(device)

    acc_pred = model(u_i, y_j)
    acc_matrix = torch.abs(acc_pred - u_j).view(B, B)
    acc_matrix.fill_diagonal_(0.0)
    acc_loss = acc_matrix.sum() / (B * (B - 1))

    # Repulsion term
    y_k = y_batch.view(1, B).expand(B, B)
    u_i_mat = u_batch.view(B, 1).expand(B, B)

    T_out = model(
        u_i_mat.reshape(-1),
        y_k.reshape(-1),
    ).view(B, B)

    rep_k = []
    for k in range(B):
        rep_k.append(mean_abs_pairwise_torch(T_out[:, k]))
    repulsion = torch.stack(rep_k).mean() * (B / (B - 1))

    return 2.0 * acc_loss - repulsion

def get_loss_fn(loss_mode: str):
    if loss_mode == "joint":
        return energy_distance_loss
    if loss_mode == "prior":
        return energy_distance_loss_fromprior
    raise ValueError(f"Unknown loss_mode: {loss_mode}")

# ============================================================
# TRUE metric: avg_{y~kappa} ED(P_theta(.|y), pi(.|y))
# ============================================================
def mean_abs_pairwise_empirical(x):
    x = np.sort(np.asarray(x).reshape(-1))
    n = x.size
    k = np.arange(1, n + 1)
    pair_sum = np.sum(x * (2*k - n - 1))
    return (2.0 * pair_sum) / (n * n)

def expected_abs_to_discrete(x, u, cum_w, cum_uw):
    x = np.asarray(x).reshape(-1)
    idx = np.searchsorted(u, x, side="left")
    Wl = np.where(idx > 0, cum_w[idx-1], 0.0)
    Ul = np.where(idx > 0, cum_uw[idx-1], 0.0)
    Wt = cum_w[-1]
    Ut = cum_uw[-1]
    Wr = Wt - Wl
    Ur = Ut - Ul
    return (x*Wl - Ul) + (Ur - x*Wr)

def EUU_discrete(u, w):
    cum_w = np.cumsum(w)
    cum_uw = np.cumsum(w*u)
    W_prev = np.concatenate([[0.0], cum_w[:-1]])
    U_prev = np.concatenate([[0.0], cum_uw[:-1]])
    return 2.0 * np.sum(w * (u*W_prev - U_prev))

def avg_energy_distance_over_y_fast(uP_samples, y_samples, G, m0, sigma0, sigma,
                                   u_span=20.0, n_grid=4001):
    y_samples = np.asarray(y_samples).reshape(-1)
    Ny = y_samples.size
    uP_samples = np.asarray(uP_samples)
    assert uP_samples.shape[0] == Ny

    a = m0 - u_span * sigma0
    b = m0 + u_span * sigma0
    u_grid = np.linspace(a, b, n_grid)
    du = u_grid[1] - u_grid[0]
    quad_w = np.ones_like(u_grid) * du
    quad_w[0] *= 0.5
    quad_w[-1] *= 0.5

    eds = np.zeros(Ny, dtype=float)

    for j, yj in enumerate(y_samples):
        x = np.asarray(uP_samples[j]).reshape(-1)

        logp = -0.5*((u_grid - m0)/sigma0)**2 - 0.5*((yj - G(u_grid))/sigma)**2
        logp -= np.max(logp)
        un = np.exp(logp)

        Z = np.sum(un * quad_w)
        w = (un / Z) * quad_w

        cum_w = np.cumsum(w)
        cum_uw = np.cumsum(w*u_grid)

        EXU = np.mean(expected_abs_to_discrete(x, u_grid, cum_w, cum_uw))
        EXX = mean_abs_pairwise_empirical(x)
        EUU = EUU_discrete(u_grid, w)

        eds[j] = 2.0*EXU - EXX - EUU

    return float(np.mean(eds)), eds

# -------------------------
# PUSHFORWARD sampler (BATCHED)
# -------------------------
@torch.no_grad()
def pushforward_samples_for_y_batched(
    model, loss_mode,
    u_base, y_base, y_dag,
    Np=5000, batch=1024, device=device
):
    """
    For loss_mode="joint": sample (u_i, y_i) from base and compute T(u_i, y_i; y_dag)
    For loss_mode="prior": sample u_i from base and compute T(u_i; y_dag)
    """
    n = len(u_base)
    idx = np.random.randint(0, n, size=Np)
    out = np.empty((Np,), dtype=np.float32)

    for start in range(0, Np, batch):
        end = min(Np, start + batch)
        sl = idx[start:end]

        u_i = torch.tensor(u_base[sl], dtype=torch.float32, device=device)

        if loss_mode == "joint":
            y_i = torch.tensor(y_base[sl], dtype=torch.float32, device=device)
            y_d = torch.full((end - start,), float(y_dag), dtype=torch.float32, device=device)
            out[start:end] = model(u_i, y_i, y_d).detach().cpu().numpy().astype(np.float32)
        else:
            y_d = torch.full((end - start,), float(y_dag), dtype=torch.float32, device=device)
            out[start:end] = model(u_i, y_d).detach().cpu().numpy().astype(np.float32)

    return out

def evaluate_ckpt_avgED_on_split(
    ckpt_path,
    y_kappa_samples,
    u_base, y_base,
    G, m0, sigma0, sigma,
    Np=5000, push_batch=1024,
    u_span=20.0, n_grid=4001
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w, d = ckpt["width"], ckpt["depth"]
    loss_mode = ckpt["loss_mode"]

    model = MLP_T(w, d, mode=loss_mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    Ny = len(y_kappa_samples)
    uP = np.stack(
        [
            pushforward_samples_for_y_batched(
                model, loss_mode=loss_mode,
                u_base=u_base, y_base=y_base, y_dag=float(y_kappa_samples[j]),
                Np=Np, batch=push_batch, device=device
            )
            for j in range(Ny)
        ],
        axis=0
    )

    avgED, _ = avg_energy_distance_over_y_fast(
        uP, y_kappa_samples, G, m0, sigma0, sigma,
        u_span=u_span, n_grid=n_grid
    )
    return float(avgED)

# -------------------------
# Train one run
# -------------------------
def train_one_config_data(
    loss_mode,
    u_train_pool, y_train_pool,
    u_val, y_val,
    K_train, hidden_dim, num_layers, seed,
    ckpt_root_dir,
    batch_size=128, lr=1e-3,
    max_epochs=100, patience=50
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_fn = get_loss_fn(loss_mode)

    # select K_train subset reproducibly
    rng = np.random.default_rng(seed)
    n_full = len(u_train_pool)
    if K_train <= n_full:
        idx = rng.choice(n_full, size=K_train, replace=False)
    else:
        idx = rng.choice(n_full, size=K_train, replace=True)
    u_tr = u_train_pool[idx]
    y_tr = y_train_pool[idx]

    train_loader = torch.utils.data.DataLoader(
        IndexedJointDataset(u_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        IndexedJointDataset(u_val, y_val),
        batch_size=min(batch_size, len(u_val)),
        shuffle=False
    )

    model = MLP_T(hidden_dim, num_layers, mode=loss_mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_curve, val_curve = [], []
    best_val, best_epoch, bad = float("inf"), -1, 0

    ckpt_dir = os.path.join(ckpt_root_dir, loss_mode)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(
        ckpt_dir, f"best_{loss_mode}_w{hidden_dim}_d{num_layers}_K{K_train}_seed{seed}.pt"
    )

    for epoch in range(max_epochs):
        model.train()
        tl = 0.0
        for u_b, y_b, _ in train_loader:
            u_b, y_b = u_b.to(device), y_b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, u_b, y_b)
            loss.backward()
            opt.step()
            tl += loss.item()
        train_curve.append(float(tl / len(train_loader)))

        model.eval()
        with torch.no_grad():
            vl = 0.0
            nb = 0
            for u_b, y_b, _ in val_loader:
                u_b, y_b = u_b.to(device), y_b.to(device)
                vl += loss_fn(model, u_b, y_b).item()
                nb += 1
        vl = float(vl / max(nb, 1))
        val_curve.append(vl)

        if vl < best_val:
            best_val, best_epoch, bad = vl, epoch, 0
            torch.save({
                "loss_mode": str(loss_mode),
                "K_train": int(K_train),
                "width": int(hidden_dim),
                "depth": int(num_layers),
                "seed": int(seed),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "n_params": int(count_params(model)),
                "model_state_dict": model.state_dict(),
                "train_curve": train_curve,
                "val_curve": val_curve,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    return {
        "loss_mode": str(loss_mode),
        "K_train": int(K_train),
        "width": int(hidden_dim),
        "depth": int(num_layers),
        "seed": int(seed),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "ckpt_path": ckpt_path,
    }

# -------------------------
# Run PARAMETER scaling sweep
# -------------------------
def run_models_param_scaling_with_avgED(
    loss_mode,
    u_train_pool, y_train_pool,
    u_val, y_val,
    u_test, y_test,
    model_configs, K_trains, seeds,
    y_kappa_val, y_kappa_test,
    ckpt_root_dir,
    batch_size=128, lr=1e-3,
    max_epochs=100, patience=50,
    Np_push=5000, push_batch=1024,
    G=lambda u: u**2, m0=0.0, sigma0=1.0, sigma=1.0,
    u_span=20.0, n_grid=4001,
    val_base="val",
    test_base="test",
):
    results = []

    # choose bases for pushforward sampling
    if val_base == "val":
        u_base_val, y_base_val = u_val, y_val
    elif val_base == "test":
        u_base_val, y_base_val = u_test, y_test
    else:
        raise ValueError("val_base must be 'val' or 'test'")

    if test_base == "test":
        u_base_test, y_base_test = u_test, y_test
    elif test_base == "val":
        u_base_test, y_base_test = u_val, y_val
    else:
        raise ValueError("test_base must be 'val' or 'test'")

    for K in K_trains:
        for cfg in model_configs:
            hidden_dim = int(cfg["hidden_dim"])
            num_layers = int(cfg["num_layers"])

            tmp = MLP_T(hidden_dim, num_layers, mode=loss_mode)
            n_params = int(count_params(tmp))
            del tmp

            ed_val_runs, ed_test_runs = [], []
            runs = []

            for s in seeds:
                out = train_one_config_data(
                    loss_mode=loss_mode,
                    u_train_pool=u_train_pool, y_train_pool=y_train_pool,
                    u_val=u_val, y_val=y_val,
                    K_train=K, hidden_dim=hidden_dim, num_layers=num_layers, seed=s,
                    ckpt_root_dir=ckpt_root_dir,
                    batch_size=batch_size, lr=lr, max_epochs=max_epochs, patience=patience
                )
                runs.append(out)

                avgED_val = evaluate_ckpt_avgED_on_split(
                    out["ckpt_path"],
                    y_kappa_samples=y_kappa_val,
                    u_base=u_base_val, y_base=y_base_val,
                    G=G, m0=m0, sigma0=sigma0, sigma=sigma,
                    Np=Np_push, push_batch=push_batch,
                    u_span=u_span, n_grid=n_grid
                )

                avgED_test = evaluate_ckpt_avgED_on_split(
                    out["ckpt_path"],
                    y_kappa_samples=y_kappa_test,
                    u_base=u_base_test, y_base=y_base_test,
                    G=G, m0=m0, sigma0=sigma0, sigma=sigma,
                    Np=Np_push, push_batch=push_batch,
                    u_span=u_span, n_grid=n_grid
                )

                ed_val_runs.append(avgED_val)
                ed_test_runs.append(avgED_test)

            rec = {
                "loss_mode": str(loss_mode),
                "K_train": int(K),
                "width": int(hidden_dim),
                "depth": int(num_layers),
                "n_params": int(n_params),

                "avgED_val_mean": float(np.mean(ed_val_runs)),
                "avgED_val_std":  float(np.std(ed_val_runs)),
                "avgED_val_runs": [float(v) for v in ed_val_runs],

                "avgED_test_mean": float(np.mean(ed_test_runs)),
                "avgED_test_std":  float(np.std(ed_test_runs)),
                "avgED_test_runs": [float(v) for v in ed_test_runs],

                "runs": runs,

                "Ny_kappa": int(len(y_kappa_val)),
                "Np_push": int(Np_push),
                "push_batch": int(push_batch),
                "u_span": float(u_span),
                "n_grid": int(n_grid),
                "G": "u**2",
                "val_base": str(val_base),
                "test_base": str(test_base),
            }
            results.append(rec)

            print(
                f"[{loss_mode}] (K={K}, w={hidden_dim}, d={num_layers}, n_params={n_params}) "
                f"avgED_val={rec['avgED_val_mean']:.4e}  avgED_test={rec['avgED_test_mean']:.4e}"
            )

    return results

# -------------------------
# Plotting: x-axis = n_params
# -------------------------
YLABEL = r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(\cdot)\right)\right]$"

import matplotlib.cm as cm

def plot_models_param_scaling_two_losses(all_results_by_loss, split="test", logx=True, logy=True):
    """
    x-axis = number of parameters
    y-axis = avgED metric

    Same color = same K_train
    Solid = joint
    Dashed = prior
    """
    assert split in ("val", "test")
    key_mean = f"avgED_{split}_mean"
    key_std  = f"avgED_{split}_std"

    fig = plt.figure()

    # Collect all unique K values
    unique_Ks = set()
    for results in all_results_by_loss.values():
        for r in results:
            unique_Ks.add(int(r["K_train"]))
    unique_Ks = sorted(unique_Ks)

    cmap = cm.get_cmap("tab10", len(unique_Ks))
    color_map = {K: cmap(i) for i, K in enumerate(unique_Ks)}

    for loss_mode in sorted(all_results_by_loss.keys()):
        results = all_results_by_loss[loss_mode]
        linestyle = LINESTYLE.get(loss_mode, "-")

        # group by K_train
        groups = {}
        for r in results:
            K = int(r["K_train"])
            groups.setdefault(K, []).append(r)

        for K in groups:
            groups[K] = sorted(groups[K], key=lambda rr: rr["n_params"])

        for K, rows in sorted(groups.items()):
            xs = [rr["n_params"] for rr in rows]
            ys = [rr[key_mean]   for rr in rows]
            es = [rr[key_std]    for rr in rows]

            plt.errorbar(
                xs, ys,
                yerr=es,
                marker="o",
                capsize=3,
                linestyle=linestyle,
                color=color_map[K],
                label=f"{loss_mode}: K={K}"
            )

    plt.xlabel("Number of parameters")
    plt.ylabel(YLABEL)
    plt.grid(True, which="both")

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.legend(frameon=False)
    plt.tight_layout()

    fname = f"scaling_models_avgED_vs_nparams_{split.upper()}_TWO_LOSSES.png"
    outpath = os.path.abspath(os.path.join(PLOT_DIR, fname))
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print("Saved figure to:", outpath)

# -------------------------
# Main
# -------------------------
def main():
    # load data
    u = np.load(U_PATH).reshape(-1)
    y = np.load(Y_PATH).reshape(-1)
    N = len(u)
    assert len(y) == N

    # fixed size splits
    if N < (VAL_SIZE + TEST_SIZE + 10):
        raise RuntimeError(f"Dataset too small (N={N}) for VAL={VAL_SIZE} and TEST={TEST_SIZE}.")

    # 1) split off TEST
    u_rest, u_test, y_rest, y_test = train_test_split(
        u, y, test_size=TEST_SIZE, random_state=0, shuffle=True
    )

    # 2) split off VAL from remaining
    u_train_pool, u_val, y_train_pool, y_val = train_test_split(
        u_rest, y_rest, test_size=VAL_SIZE, random_state=1, shuffle=True
    )

    # fixed y~kappa for VAL and TEST
    rng = np.random.default_rng(0)

    def sample_kappa(y_arr, Ny):
        y_arr = np.asarray(y_arr).reshape(-1)
        if len(y_arr) >= Ny:
            return rng.choice(y_arr, size=Ny, replace=False)
        else:
            return rng.choice(y_arr, size=Ny, replace=True)

    y_kappa_val  = sample_kappa(y_val,  NY_KAPPA)
    y_kappa_test = sample_kappa(y_test, NY_KAPPA)

    np.save(os.path.join(OUT_DIR, "y_kappa_val.npy"), y_kappa_val)
    np.save(os.path.join(OUT_DIR, "y_kappa_test.npy"), y_kappa_test)

    # meta
    meta = dict(
        run_tag=RUN_TAG,
        data_dir=DATA_DIR,
        device=str(device),
        loss_modes=LOSS_MODES,
        model_configs=MODEL_CONFIGS,
        K_trains=[int(k) for k in K_TRAINS],
        seeds=[int(s) for s in SEEDS],
        batch_size=int(BATCH_SIZE),
        lr=float(LR),
        max_epochs=int(MAX_EPOCHS),
        patience=int(PATIENCE),
        Ny_kappa=int(NY_KAPPA),
        Np_push=int(NP_PUSH),
        push_batch=int(PUSH_BATCH),
        u_span=float(U_SPAN),
        n_grid=int(N_GRID),
        m0=float(M0),
        sigma0=float(SIGMA0),
        sigma=float(SIGMA),
        G="u**2",
        sizes=dict(
            train_pool=int(len(u_train_pool)),
            val=int(len(u_val)),
            test=int(len(u_test)),
            total=int(N),
        )
    )
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # run sweeps for BOTH losses
    all_results_by_loss = {}
    for loss_mode in LOSS_MODES:
        results = run_models_param_scaling_with_avgED(
            loss_mode=loss_mode,
            u_train_pool=u_train_pool, y_train_pool=y_train_pool,
            u_val=u_val, y_val=y_val,
            u_test=u_test, y_test=y_test,
            model_configs=MODEL_CONFIGS,
            K_trains=K_TRAINS, seeds=SEEDS,
            y_kappa_val=y_kappa_val, y_kappa_test=y_kappa_test,
            ckpt_root_dir=CKPT_DIR,
            batch_size=BATCH_SIZE, lr=LR,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
            Np_push=NP_PUSH, push_batch=PUSH_BATCH,
            G=G, m0=M0, sigma0=SIGMA0, sigma=SIGMA,
            u_span=U_SPAN, n_grid=N_GRID,
            val_base="val",
            test_base="test",
        )
        all_results_by_loss[loss_mode] = results

    # save combined results
    combined_path = os.path.join(OUT_DIR, "results_models_param_scaling_two_losses.json")
    with open(combined_path, "w") as f:
        json.dump(all_results_by_loss, f, indent=2)

    # plots
    plot_models_param_scaling_two_losses(all_results_by_loss, split="val",  logx=True, logy=True)
    plot_models_param_scaling_two_losses(all_results_by_loss, split="test", logx=True, logy=True)

    print("\nSaved to:", os.path.abspath(OUT_DIR))
    print("results:", combined_path)
    print("checkpoints:", CKPT_DIR)
    print("plots:", PLOT_DIR)

if __name__ == "__main__":
    main()