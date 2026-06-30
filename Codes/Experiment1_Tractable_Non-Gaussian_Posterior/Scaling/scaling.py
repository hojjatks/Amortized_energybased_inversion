# ============================================================
# Scaling study: sweep width + depth, train each model, and SAVE
# IMPORTANT OUTPUTS so you don't retrain:
#   - save best checkpoint per (width, depth, seed) as .pt
#   - save aggregated results.json (NO numpy arrays inside)
#   - save plots
# ============================================================

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split

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
u = np.load(os.path.join(data_dir, "samples_u.npy"))
y = np.load(os.path.join(data_dir, "samples_y.npy"))

# Consistent splits
u_train_test, u_infer, y_train_test, y_infer = train_test_split(
    u, y, test_size=0.5, random_state=0
)
u_train, u_val, y_train, y_val = train_test_split(
    u_train_test, y_train_test, test_size=0.1, random_state=0
)

# Subset
K_train, K_val = 50000, 1000
max_epochs = 100
u_train, y_train = u_train[:K_train], y_train[:K_train]
u_val, y_val     = u_val[:K_val],     y_val[:K_val]
print(f"Subset sizes → train={len(u_train)}, val={len(u_val)}")

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
# Energy distance loss
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
        layers = [nn.Linear(3, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
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

            # save best checkpoint
            torch.save({
                "width": int(hidden_dim),
                "depth": int(num_layers),
                "seed": int(seed),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "n_params": int(count_params(model)),
                "model_state_dict": model.state_dict(),
                "train_curve": train_curve,  # plain lists
                "val_curve": val_curve,      # plain lists
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

# -------------------------
# Sweep
# -------------------------
def run_sweep(widths, depths, seeds,
              batch_size=512, lr=1e-3,
              max_epochs=100, patience=50):

    results = []

    for w, d in product(widths, depths):
        runs = []
        for s in seeds:
            out = train_one_config(
                u_train, y_train, u_val, y_val,
                hidden_dim=w, num_layers=d, seed=s,
                batch_size=batch_size, lr=lr,
                max_epochs=max_epochs, patience=patience
            )
            runs.append(out)

        # param count (compute once for this (w,d))
        tmp = MLP_T(w, d)
        n_params = int(count_params(tmp))
        del tmp

        best_vals = [r["best_val"] for r in runs]
        rec = {
            "width": int(w),
            "depth": int(d),
            "n_params": int(n_params),
            "val_mean": float(np.mean(best_vals)),
            "val_std": float(np.std(best_vals)),
            "runs": runs,  # NO numpy arrays in here
        }
        results.append(rec)

        print(f"(w={w}, d={d}) params={n_params:,}  val_mean={rec['val_mean']:.4e}")

        # save after each (w,d) so you can resume if job dies
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results

# -------------------------
# Helpers for plots
# -------------------------
def _pivot_val_mean(results):
    widths = sorted({r["width"] for r in results})
    depths = sorted({r["depth"] for r in results})
    val_mean = {(r["width"], r["depth"]): r["val_mean"] for r in results}
    val_std  = {(r["width"], r["depth"]): r["val_std"]  for r in results}
    return widths, depths, val_mean, val_std

def plot_scaling_by_width(results, logy=True):
    widths, depths, val_mean, val_std = _pivot_val_mean(results)

    plt.figure()
    for d in depths:
        ys = [val_mean[(w, d)] for w in widths]
        es = [val_std[(w, d)]  for w in widths]
        plt.errorbar(widths, ys, yerr=es, marker="o", capsize=3, label=f"depth={d}")

    plt.xlabel("Width (hidden_dim)")
    plt.ylabel("Validation loss (mean over seeds)")
    plt.title("val loss vs width (one line per depth)")
    plt.grid(True, which="both")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_val_vs_width_by_depth.png"), dpi=300)
    plt.show()

def plot_scaling_by_depth(results, logy=True):
    widths, depths, val_mean, val_std = _pivot_val_mean(results)

    plt.figure()
    for w in widths:
        ys = [val_mean[(w, d)] for d in depths]
        es = [val_std[(w, d)]  for d in depths]
        plt.errorbar(depths, ys, yerr=es, marker="o", capsize=3, label=f"width={w}")

    plt.xlabel("Depth (num_layers)")
    plt.ylabel("Validation loss (mean over seeds)")
    plt.title("val loss vs depth (one line per width)")
    plt.grid(True, which="both")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scaling_val_vs_depth_by_width.png"), dpi=300)
    plt.show()

def plot_learning_curves_from_ckpt(results):
    # load curves from each saved checkpoint, plot solid=train dashed=val
    for r in results:
        w, d = r["width"], r["depth"]
        plt.figure()
        for i, run in enumerate(r["runs"]):
            ckpt = torch.load(run["ckpt_path"], map_location="cpu")
            tr = ckpt["train_curve"]
            va = ckpt["val_curve"]

            ln, = plt.plot(tr, linestyle="-", alpha=0.7, label="train" if i == 0 else None)
            c = ln.get_color()
            plt.plot(va, linestyle="--", color=c, alpha=0.7, label="val" if i == 0 else None)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"w={w}, d={d} (N={r['n_params']:,})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"learningcurve_w{w}_d{d}.png"), dpi=300)
        plt.show()

# ============================================================
# RUN
# ============================================================
widths = (4, 8, 16)
depths = (1, 2, 4)
seeds  = (0, 1, 2)

results = run_sweep(widths, depths, seeds, batch_size=512, lr=1e-3, max_epochs=max_epochs, patience=50)

plot_scaling_by_width(results, logy=True)
plot_scaling_by_depth(results, logy=True)
plot_learning_curves_from_ckpt(results)

print("\nSaved to:", OUT_DIR)
print("results:", os.path.join(OUT_DIR, "results.json"))
print("checkpoints:", CKPT_DIR)
print("plots:", PLOT_DIR)
