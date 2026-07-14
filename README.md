# Amortized Energy Distance-Based Operator Learning for Infinite-Dimensional Bayesian Inverse Problems

**Authors:** R. Baptista, H. Kaveh, A. M. Stuart

This repository provides the code and data needed to reproduce the numerical results in the paper:

> R. Baptista, H. Kaveh, A. M. Stuart, "Amortized Energy Distance-Based Operator Learning for Infinite-Dimensional Bayesian Inverse Problems," [arXiv:2605.15407](https://arxiv.org/abs/2605.15407).

It is provided for the purpose of satisfying the *SIAM Journal on Scientific Computing* Reproducibility of Computational Results policy (the "SISC Reproducibility Badge"). Every figure and table in the paper can be regenerated from the material here; see the [Figure and table reproduction guide](#figure-and-table-reproduction-guide) below for an exact mapping from paper result to script.

## Repository layout

```
Codes/
  Experiment1_Tractable_Non-Gaussian_Posterior/   # Sec. 3.1: finite-dim. toy problem (MLP)
  Experiment2_Darcy Flow Inverse Problem/          # Sec. 3.2: 1D Darcy flow (FNO)
  Experiment3_Wave Equation Inverse Problem/       # Sec. 3.3: 1D wave equation (FNO)
Data/          # generated training/inference data + pCN chains (see Data & Models availability)
Models/        # trained model checkpoints (see Data & Models availability)
Figs/          # figures written by the scripts/notebooks below (regenerated, not stored in git)
env.yml        # conda environment specification
```

**Note on internal paths:** the code was developed with a different, private experiment-numbering scheme before the `Codes/` folders were renamed for public release. Internally, scripts for the paper's "Experiment 2" (Darcy) still read/write `Data/Experiment3/...` and `Figs/Experiment3/...`, and scripts for "Experiment 3" (Wave) read/write `Data/Experiment7/...`, `Models/Experiment7/...`, `Figs/Experiment7/...`. This is intentional — do not rename these subfolders, or the notebooks will fail to find their inputs.

## Environment setup

**Python** (Experiments 1–3, all model training/inference/plotting):
```
conda env create -f env.yml
conda activate MNM
```
This installs Python 3.10, PyTorch 2.2 (CUDA 12.1), NumPy, SciPy, scikit-learn, seaborn, h5py, JupyterLab, and `torch_dct` (the discrete-cosine-transform library used by the Fourier neural operator in Experiments 2–3).

Two optional diagnostic dependencies are used only by non-canonical exploratory notebooks (not required to reproduce any paper figure): `psutil` (GPU/CPU memory logging in `Helper.py`) and `arviz` (MCMC effective-sample-size diagnostics in `GenData/check_pCN.ipynb`, Experiment 3).

**MATLAB** (Experiment 2 data generation only): `Codes/Experiment2_Darcy Flow Inverse Problem/GenData/*.m` requires MATLAB with the base Signal Processing functionality for `idct`. Any recent MATLAB release (R2020a or later) should work.



## Data & Models availability

`Data/` (~20GB) and `Models/` (~160MB) are excluded from git (see `.gitignore`) and are instead distributed via:
- **Zenodo: [10.5281/zenodo.21364643](https://doi.org/10.5281/zenodo.21364643)**, and
- the zipped snapshot deposited in the SISC Supplementary Materials for this paper.

The Zenodo record contains four archives — `Data_Experiment1.zip`, `Data_Experiment3_Darcy.zip`, `Data_Experiment7_Wave.zip`, `Models.zip` — matching the folders described below. To reproduce results, download and extract them so that `Data/` and `Models/` sit at the repository root, as siblings of `Codes/` (matching the layout shown above).

| Folder | Contents | Used by |
|---|---|---|
| `Data/Experiment1/` | `samples_u.npy`, `samples_y.npy` — 2M prior/joint samples for the toy problem | Experiment 1 |
| `Data/Experiment3/` | `darcy_data1D_64.mat` (1M Darcy joint samples), pCN chains, pushforward samples, Wasserstein results | Experiment 2 (Darcy) |
| `Data/Experiment7/` | `samples_wave_center.npz` (2M wave-equation joint samples), pCN chains, wave-iteration truths, Wasserstein results | Experiment 3 (Wave) |
| `Models/fromprior_CosineFNO_cmON_..._641D_20260601_0524.pth` | Trained Darcy FNO (Cameron–Martin on) | Experiment 2 inference/plotting |
| `Models/Experiment7/` | Trained Wave FNOs (CM on/off) | Experiment 3 inference/plotting |

If you would rather regenerate the data than download it, the generation scripts are documented per-experiment below — but note the caveats in [Known limitations](#known-limitations--non-canonical-code) regarding non-determinism and runtime.

## Figure and table reproduction guide

All figure numbers below are inferred from the order figures appear in the paper. 

### Table 1 — Training and model settings

Reproduced directly by the hyperparameters hard-coded in the training scripts listed below; no separate script generates the table.

### Experiment 1 — Tractable Non-Gaussian Posterior (Sec. 3.1)

| Result | Script | Notes |
|---|---|---|
| Fig. 1 (`fig:Exp1_post`, `four_posteriors_two_rows_prior.png`) | `Codes/Experiment1_Tractable_Non-Gaussian_Posterior/GenData.py` → `ML.ipynb` | Data: `GenData.py` (`np.random.seed(42)`, N=2,000,000). Training: `ML.ipynb` with `fromjoint = False` (the "prior" pushforward map, `run_tag="prior"`), `ndata=600000`, MLP width 150 / depth 8, batch 1000, 60 epochs — matches Table 1 column 1 exactly. Run the `plot_four_posteriors_two_rows(..., y_values=(-1,0,1,2))` cell for Fig. 1. |
| Fig. 5 (`fig:scaling_data`, Appendix D) | `Codes/Experiment1_Tractable_Non-Gaussian_Posterior/Scaling/scaling_models_data_two_losses.py`, replotted by `Scaling/replot_two_losses.py` | Sweeps training-set size `K_TRAINS` for both the "joint" and "prior" reference-measure choices; saves `results_models_data_scaling_two_losses.json` and `plots/scaling_models_avgED_vs_dataK_TEST_TWO_LOSSES.png`, matching the figure exactly. |

### Experiment 2 — Darcy Flow Inverse Problem (Sec. 3.2)

| Result | Script | Notes |
|---|---|---|
| Data generation | `Codes/Experiment2_Darcy Flow Inverse Problem/GenData/GenData.m` (MATLAB) | Draws N=1,000,000 log-permeability fields (`gaussrnd.m`, τ=3, α=2), solves the 1D Darcy equation on a 64-point grid (`solve_gwf_1D.m`), saves `Data/Experiment3/darcy_data1D_64.mat`. **Not currently seeded** — see limitations. |
| pCN reference posteriors | `Codes/Experiment2_Darcy Flow Inverse Problem/pCN/pCN.ipynb` (and `outputs_pCN_it-1.ipynb`, `outputs_pCN_it-3.ipynb` for the other two observations) | Runs pCN for 10⁶ steps with 2×10⁵ burn-in, as stated in the paper. Three independent runs at `index_data ∈ {0, -1, -3}` produce the three rows of Fig. 2. (`outputs_pCN_it-2.ipynb` is an extra run not used in the final figure.) |
| Model training | `Codes/Experiment2_Darcy Flow Inverse Problem/ML_FNO.ipynb` or `Inference.ipynb` | FNO width 80, depth 5, modes 32, batch 150, 60 epochs, patience/trigger 20 — matches Table 1 column 2. Saves checkpoint to `Models/`. |
| Fig. 2 (`fig:Exp2_phys`, physical-space posterior, 3 rows) | `Codes/Experiment2_Darcy Flow Inverse Problem/Inference.ipynb` | Rerun with `iteration_data ∈ {0, -1, -3}` (must match the pCN runs above); saves `Figs/Experiment3/physical_space_with_obs...+iteration{N}.png`. |
| Fig. 3 (`fig:Exp2_project2`, KL-mode comparison) | `Codes/Experiment2_Darcy Flow Inverse Problem/PlotWass.ipynb` | Saves `Figs/Experiment3/modes_grid_T_prior_bestCMON.png`. |
| Fig. 4 (`fig:Exp2_wass`, Cameron–Martin ablation, per-mode Wasserstein) | `Codes/Experiment2_Darcy Flow Inverse Problem/PlotWass.ipynb` (second half, loads both CM-on and CM-off Wasserstein `.npz` files) | Renamed `CMeffect.png` for the manuscript. |

### Experiment 3 — Wave Equation Inverse Problem (Sec. 3.3)

| Result | Script | Notes |
|---|---|---|
| Data generation | `Codes/Experiment3_Wave Equation Inverse Problem/GenData/sample_joint.py` | Draws N=2,000,000 latent Gaussian fields (τ=5, α=2, scale=10), thresholds to a binary wavespeed field, solves the 1D wave equation (`forwardmodel.py`, leapfrog FD), extracts first-arrival times. `np.random.seed(1)` — deterministic. Saves `Data/Experiment7/samples_wave_center.npz`. |
| pCN reference posteriors | `Codes/Experiment3_Wave Equation Inverse Problem/GenData/pCN.ipynb` | Runs pCN for 2.5×10⁶ steps with 1×10⁵ burn-in, as stated in the paper (`seed=1373`, deterministic). Rerun per observation instance (`iteration_true`/`iteration_data` ∈ {-6, -4, -1}) for the three rows of Fig. 7. |
| Fig. 6 (`fig:exp3_spatiotemporal`, wavefield) | `Codes/Experiment3_Wave Equation Inverse Problem/GenData/PlotForwardmodel.ipynb` (`iteration_data = -6`) | Saves `Figs/Experiment7/wavefield_iteration-6.png`. |
| Model training | `Codes/Experiment3_Wave Equation Inverse Problem/ML_FNO.ipynb` | FNO width 100, depth 5, modes 32, batch 80, 15 epochs, `learning_ratio=9/12` (→ 1.5M training samples) — matches Table 1 column 3 exactly. |
| Fig. 7 (`fig:exp3_posterior`, wavespeed reconstruction, 3 rows) | `Codes/Experiment3_Wave Equation Inverse Problem/PlotIterationsLatent.ipynb` | `iterations_to_load = [-6, -4, -1]`; saves `Figs/Experiment7/wavespeed_latent_all_iterations_scaled.png`. **Use this notebook, not `PlotIterations.ipynb`** (see limitations). |
| Fig. 8 (`fig:exp3_modes`, KL-mode comparison) | `Codes/Experiment3_Wave Equation Inverse Problem/Inference.ipynb` | With `use_cm=True`, `iteration_data=-6`; saves `Figs/Experiment7/modes_grid_T_CM_ONCMTrue_iteration--6.png`. |
| Fig. 9 (`fig:exp3_wass`, Cameron–Martin ablation) | `Codes/Experiment3_Wave Equation Inverse Problem/PlotWass.ipynb` | Renamed `Exp3_wasserstein_per_mode.png` for the manuscript. |


## License

Code is released under the MIT License (see `LICENSE`).
