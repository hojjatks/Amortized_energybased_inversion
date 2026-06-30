# ============================================================
# Scaling study: sweep width+depth, train each model, SAVE
# Then evaluate: avg_{y~kappa} EnergyDistance( P_theta(.|y), pi(.|y) )
# using 30 y samples from kappa (taken from y_infer).
#
# Here: TRUE forward model is G(u) = u^2.
# ============================================================


# ------------------------
# helper functions
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def plot_scaling_by_params_avgED(results, logx=True, logy=True):
    plt.figure()

    # group by depth (one line per depth)
    depths = sorted({r["depth"] for r in results})
    for d in depths:
        subset = sorted(
            [r for r in results if r["depth"] == d],
            key=lambda r: r["n_params"]
        )
        xs = [r["n_params"] for r in subset]
        ys = [r["avgED_mean"] for r in subset]
        es = [r["avgED_std"]  for r in subset]

        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"depth={d}")

    plt.xlabel("Number of parameters")
    plt.ylabel(r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(\cdot)\right)\right]$")
    # plt.title("avg_y ED vs number of parameters (one line per depth)")
    plt.grid(True, which="both")

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_avgED_vs_params_by_depth.png"), dpi=300)
    plt.show()


def plot_scaling_by_params_avgED_by_width(results, logx=True, logy=True):
    plt.figure()

    # group by width (one line per width)
    widths = sorted({r["width"] for r in results})
    for w in widths:
        subset = sorted(
            [r for r in results if r["width"] == w],
            key=lambda r: r["n_params"]
        )
        xs = [r["n_params"] for r in subset]
        ys = [r["avgED_mean"] for r in subset]
        es = [r["avgED_std"]  for r in subset]

        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"width={w}")

    plt.xlabel("Number of parameters")
    plt.ylabel(r"$\mathbb{E}_{y\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(.)\right)\right]$")
    # plt.title("avg_y ED vs number of parameters (one line per width)")
    plt.grid(True, which="both")

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_avgED_vs_params_by_width.png"), dpi=300)
    plt.show()


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
# Energy distance LOSS (training)
# -------------------------
def energy_distance_loss(model, u_batch, y_batch):
    B = u_batch.shape[0]

    u_i = u_batch.unsqueeze(1).expand(B, B).reshape(-1)
    y_i = y_batch.unsqueeze(1).expand(B, B).reshape(-1)
    y_j = y_batch.unsqueeze(0).expand(B, B).reshape(-1)
    u_j = u_batch.unsqueeze(0).expand(B, B).reshape(-1)

    acc_pred = model(u_i, y_i, y_j)
    acc_loss = torch.mean(torch.abs(acc_pred - u_j))

    y_k = y_batch.view(1, B).expand(B, B)
    u_i_mat = u_batch.view(B, 1).expand(B, B)
    y_i_mat = y_batch.view(B, 1).expand(B, B)

    T_out = model(
        u_i_mat.reshape(-1),
        y_i_mat.reshape(-1),
        y_k.reshape(-1),
    ).view(B, B)

    diffs = T_out.unsqueeze(0) - T_out.unsqueeze(1)
    dists = torch.abs(diffs)

    repulsion = dists.mean(dim=(0, 1)).mean() * (B / (B - 1))
    return 2 * acc_loss - repulsion

# ------------------------- 
# Model
# -------------------------
class MLP_T(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u, y, y_dag):
        x = torch.stack([u, y, y_dag], dim=-1)
        return u + self.net(x).squeeze(-1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# Train one run (SAVES best checkpoint)
# -------------------------
def train_one_config(u_train, y_train, u_val, y_val,
                     hidden_dim, num_layers, seed,
                     batch_size=512, lr=1e-3,
                     max_epochs=100, patience=50):

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        IndexedJointDataset(u_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        IndexedJointDataset(u_val, y_val),
        batch_size=len(u_val), shuffle=False
    )

    model = MLP_T(hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_curve = []
    val_curve   = []
    best_val = float("inf")
    best_epoch = -1
    bad = 0

    ckpt_path = os.path.join(CKPT_DIR, f"best_w{hidden_dim}_d{num_layers}_seed{seed}.pt")

    for epoch in range(max_epochs):
        model.train()
        tl = 0.0
        for u_b, y_b, _ in train_loader:
            u_b, y_b = u_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = energy_distance_loss(model, u_b, y_b)
            loss.backward()
            opt.step()
            tl += loss.item()
        train_curve.append(float(tl / len(train_loader)))

        model.eval()
        with torch.no_grad():
            vl = 0.0
            for u_b, y_b, _ in val_loader:
                u_b, y_b = u_b.to(device), y_b.to(device)
                vl += energy_distance_loss(model, u_b, y_b).item()
        vl = float(vl / len(val_loader))
        val_curve.append(vl)

        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            bad = 0

            torch.save({
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
        "seed": int(seed),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "ckpt_path": ckpt_path,
    }

# ============================================================
# TRUE metric: avg_{y~kappa} ED(P_theta(.|y), pi(.|y))
# (fast 1D quadrature ED, tested)
# ============================================================

def mean_abs_pairwise_empirical(x):
    x = np.sort(np.asarray(x).reshape(-1))
    n = x.size
    k = np.arange(1, n + 1)
    pair_sum = np.sum(x * (2*k - n - 1))  # sum_{i<j}(x_j - x_i)
    return (2.0 * pair_sum) / (n*n)

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
        w = (un / Z) * quad_w  # masses sum to 1

        cum_w = np.cumsum(w)
        cum_uw = np.cumsum(w*u_grid)

        EXU = np.mean(expected_abs_to_discrete(x, u_grid, cum_w, cum_uw))
        EXX = mean_abs_pairwise_empirical(x)
        EUU = EUU_discrete(u_grid, w)

        eds[j] = 2.0*EXU - EXX - EUU

    return float(np.mean(eds)), eds

# -------------------------
# Pushforward sampler: build P_theta(.|y_dag) from joint samples (u_infer,y_infer)
# -------------------------
@torch.no_grad()
def pushforward_samples_for_y(model, u_base, y_base, y_dag, Np=5000, device=device):
    """
    Approximate samples from P_theta(.|y_dag) by:
      pick random indices i, take (u_i, y_i), compute T(u_i, y_i; y_dag).
    """
    n = len(u_base)
    idx = np.random.randint(0, n, size=Np)
    u_i = torch.tensor(u_base[idx], dtype=torch.float32, device=device)
    y_i = torch.tensor(y_base[idx], dtype=torch.float32, device=device)
    y_d = torch.full((Np,), float(y_dag), dtype=torch.float32, device=device)
    out = model(u_i, y_i, y_d).detach().cpu().numpy()
    return out

def evaluate_ckpt_avgED(ckpt_path, y_kappa_samples, u_base, y_base,
                        G, m0, sigma0, sigma,
                        Np=5000, u_span=20.0, n_grid=4001):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w, d = ckpt["width"], ckpt["depth"]
    model = MLP_T(w, d).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    Ny = len(y_kappa_samples)
    uP = np.stack(
        [pushforward_samples_for_y(model, u_base, y_base, y_dag=y_kappa_samples[j], Np=Np)
         for j in range(Ny)],
        axis=0
    )

    avgED, _ = avg_energy_distance_over_y_fast(
        uP, y_kappa_samples, G, m0, sigma0, sigma,
        u_span=u_span, n_grid=n_grid
    )
    return float(avgED)

# -------------------------
# Sweep (training + evaluation metric)
# -------------------------
def run_sweep_with_avgED(widths, depths, seeds,
                         batch_size=512, lr=1e-3,
                         max_epochs=100, patience=50,
                         Ny_kappa=30, Np_push=5000,
                         G=lambda u: u**2, m0=0.0, sigma0=1.0, sigma=1.0,
                         u_span=20.0, n_grid=4001):

    results = []

    # draw 30 y~kappa from held-out y_infer (empirical kappa)
    rng = np.random.default_rng(0)
    if len(y_infer) >= Ny_kappa:
        y_kappa = rng.choice(y_infer, size=Ny_kappa, replace=False)
    else:
        y_kappa = rng.choice(y_infer, size=Ny_kappa, replace=True)

    for w, d in product(widths, depths):
        runs = []
        ed_runs = []

        for s in seeds:
            out = train_one_config(
                u_train, y_train, u_val, y_val,
                hidden_dim=w, num_layers=d, seed=s,
                batch_size=batch_size, lr=lr,
                max_epochs=max_epochs, patience=patience
            )
            runs.append(out)

            avgED = evaluate_ckpt_avgED(
                out["ckpt_path"],
                y_kappa_samples=y_kappa,
                u_base=u_infer, y_base=y_infer,
                G=G, m0=m0, sigma0=sigma0, sigma=sigma,
                Np=Np_push, u_span=u_span, n_grid=n_grid
            )
            ed_runs.append(avgED)

        tmp = MLP_T(w, d)
        n_params = int(count_params(tmp))
        del tmp

        rec = {
            "width": int(w),
            "depth": int(d),
            "n_params": int(n_params),
            "avgED_mean": float(np.mean(ed_runs)),
            "avgED_std": float(np.std(ed_runs)),
            "avgED_runs": [float(v) for v in ed_runs],
            "runs": runs,
            "Ny_kappa": int(Ny_kappa),
            "Np_push": int(Np_push),
            "u_span": float(u_span),
            "n_grid": int(n_grid),
            "G": "u**2",
        }
        results.append(rec)

        print(f"(w={w}, d={d}) params={n_params:,}  avgED_mean={rec['avgED_mean']:.4e}")

        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results

# -------------------------
# Plot helpers for avgED
# -------------------------
def _pivot_metric(results, key_mean="avgED_mean", key_std="avgED_std"):
    widths = sorted({r["width"] for r in results})
    depths = sorted({r["depth"] for r in results})
    mean = {(r["width"], r["depth"]): r[key_mean] for r in results}
    std  = {(r["width"], r["depth"]): r[key_std]  for r in results}
    return widths, depths, mean, std

def plot_scaling_by_width_avgED(results, logy=True):
    widths, depths, mean, std = _pivot_metric(results)

    plt.figure()
    for d in depths:
        ys = [mean[(w, d)] for w in widths]
        es = [std[(w, d)]  for w in widths]
        plt.errorbar(widths, ys, yerr=es, marker="o", capsize=3, label=f"depth={d}")

    plt.xlabel("Width (hidden_dim)")
    plt.ylabel(r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(.)\right)\right]$")
    plt.grid(True, which="both")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_avgED_vs_width_by_depth.png"), dpi=300)
    plt.show()

def plot_scaling_by_depth_avgED(results, logy=True):
    widths, depths, mean, std = _pivot_metric(results)

    plt.figure()
    for w in widths:
        ys = [mean[(w, d)] for d in depths]
        es = [std[(w, d)]  for d in depths]
        plt.errorbar(depths, ys, yerr=es, marker="o", capsize=3, label=f"width={w}")

    plt.xlabel("Depth (num_layers)")
    plt.ylabel(r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(.)\right)\right]$")
    plt.title("avg_y ED vs depth (one line per width)")
    plt.grid(True, which="both")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_avgED_vs_depth_by_width.png"), dpi=300)
    plt.show()



# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Output folders
# -------------------------
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"./scaling_saved_{RUN_TAG}"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
data_dir = "./../../../Data/Experiment1"
u = np.load(os.path.join(data_dir, "samples_u.npy")).reshape(-1)
y = np.load(os.path.join(data_dir, "samples_y.npy")).reshape(-1)

# Consistent splits
u_train_test, u_infer, y_train_test, y_infer = train_test_split(
    u, y, test_size=0.5, random_state=0
)
u_train, u_val, y_train, y_val = train_test_split(
    u_train_test, y_train_test, test_size=0.1, random_state=0
)

# Subset
K_train, K_val = 30_000, 1000
max_epochs = 50
u_train, y_train = u_train[:K_train], y_train[:K_train]
u_val, y_val     = u_val[:K_val],     y_val[:K_val]
print(f"Subset sizes → train={len(u_train)}, val={len(u_val)}; infer={len(u_infer)}")

# Save basic run info
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump({
        "run_tag": RUN_TAG,
        "data_dir": data_dir,
        "K_train": int(K_train),
        "K_val": int(K_val),
        "max_epochs": int(max_epochs),
        "device": str(device),
    }, f, indent=2)


# ============================================================
# RUN
# ============================================================
m0, sigma0, sigma = 0.0, 1.0, 1.0
G = lambda u: u**2

# widths = (4, 8, 16)
# depths = (1, 2, 4)
# seeds  = (0, 1, 2)
widths = (2, 4, 8, 16, 32, 64, 128)
depths = (2,)
seeds  = (0, 1, 2, 3)
results = run_sweep_with_avgED(
    widths, depths, seeds,
    batch_size=700, lr=1e-3,
    max_epochs=max_epochs, patience=50,
    Ny_kappa=30, Np_push=5000,
    G=G, m0=m0, sigma0=sigma0, sigma=sigma,
    u_span=20.0, n_grid=4001
)

plot_scaling_by_width_avgED(results, logy=True)
plot_scaling_by_depth_avgED(results, logy=True)
plot_scaling_by_params_avgED(results)
plot_scaling_by_params_avgED_by_width(results)

print("\nSaved to:", OUT_DIR)
print("results:", os.path.join(OUT_DIR, "results.json"))
print("checkpoints:", CKPT_DIR)
print("plots:", PLOT_DIR)
