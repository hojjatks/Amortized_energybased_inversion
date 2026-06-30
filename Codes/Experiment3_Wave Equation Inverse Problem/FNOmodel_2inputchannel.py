# multi_input_cosine_fno_neumann.py
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch_dct as dct


# ============================================
# DCT helpers (2D orthonormal on last two dims)
# ============================================

def dct1(u: torch.Tensor) -> torch.Tensor:
    """1D orthonormal DCT-II along the specified dimension."""
    return dct.dct(u, norm="ortho")

def idct1(U: torch.Tensor) -> torch.Tensor:
    """1D inverse DCT (DCT-III) with orthonormal normalization."""
    return dct.idct(U, norm="ortho")

# https://github.com/zh217/torch-dct/blob/0804f5ed2ddcaecc24c14b096bd62695e0478cec/torch_dct/_dct.py#L31




# ============================================
# Neumann eigenvalues (continuum)
# ============================================

def neumann_eigs_1d( Nx: int, Lx: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Continuum Neumann Laplacian eigenvalues on [0,Lx]:
        λ_{ℓ} = (π ℓ / Lx)^2, ℓ=0..Nx-1
    Returns tensor (Nx) on device/dtype.
    """
    kx = torch.arange(Nx, device=device, dtype=dtype)
    lam_x = (math.pi * kx / Lx) ** 2
    return lam_x


# ============================================
# Cosine spectral convolution (Neumann BC) - 1D
# ============================================

class CosineSpectralConv1d(nn.Module):
    """
    Resolution-invariant spectral convolution in the 1D cosine (Neumann) basis.

    Steps:
        1) DCT-II of input (Neumann BC).
        2) Select low-frequency modes.
        3) Apply learned linear map Cin → Cout on each mode.
        4) Zero-pad to full spectrum.
        5) Inverse DCT-II.

    modes_x can be:
      - int     → index cutoff
      - float∈(0,1] → fraction of modes (resolution-invariant)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: float | int,
                 max_modes_x: int = 32):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        # User specification
        self.modes_x_spec = float(modes_x)
        self._use_fraction_x = (0.0 < self.modes_x_spec <= 1.0) and (not self.modes_x_spec.is_integer())

        self.max_modes_x = max_modes_x

        # Weight: (Cout, Cin, M)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.max_modes_x) * 0.02
        )

    @torch.no_grad()
    def _cap_modes(self, Nx: int) -> int:
        """Determine number of active modes."""
        if self._use_fraction_x:
            mx = int(round(self.modes_x_spec * Nx))
        else:
            mx = int(self.modes_x_spec)

        mx = min(mx, Nx, self.max_modes_x)
        return max(1, mx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, C, Nx)
        Output: (B, C_out, Nx)
        """
        assert x.dim() == 3, "Expected (B, C, Nx)"
        B, Cin, Nx = x.shape
        assert Cin == self.in_channels

        # 1) DCT-II (1D)
        Xc = dct1(x)  # (B, Cin, Nx)

        # 2) Mode window
        mx = self._cap_modes(Nx)

        # 3) Slice spectrum and weights
        Xc_low = Xc[:, :, :mx]                # (B, Cin, mx)
        W      = self.weight[:, :, :mx]       # (Cout, Cin, mx)

        # 4) Per-mode linear mixing: (o,i,m) × (b,i,m) → (b,o,m)
        Yc_low = torch.einsum("oim, bim -> bom", W, Xc_low)

        # 5) Zero pad in spectral domain
        Yc = torch.zeros(B, self.out_channels, Nx,
                         dtype=x.dtype, device=x.device)
        Yc[:, :, :mx] = Yc_low

        # 6) Inverse DCT
        y = idct1(Yc)
        return y



# ============================================
# Cameron–Martin projection: C^{1/2} in 1D
# ============================================

class CMProjectionNeumann1d(nn.Module):
    """
    Resolution-invariant Cameron–Martin projection for 1D Gaussian priors
    with Neumann boundary conditions.

    Applies the operator:
        C^{1/2} = (-Δ + τ² I)^{-α/2}
    to each channel of z(x), in the cosine (Neumann) spectral basis.

    ZERO-MODE REMOVAL
    -----------------
    This implementation explicitly removes the k = 0 Neumann mode:
        û(k=0) = 0,
    which removes the constant (mean) component.

    This matches the prior sampling procedure and excludes the
    Neumann Laplacian zero eigenvalue from the Cameron–Martin space.

    Parameters
    ----------
    Lx : float
        Physical domain length.
    alpha : float
        Exponent in C = A^{-α}.
    tau : float
        Shift in A = -Δ + τ² I (τ > 0).
    min_eig : float
        Minimum eigenvalue for numerical stability.

    Input
    -----
    z : (B, C, Nx)

    Output
    ------
    u : (B, C, Nx)
        Spectrally filtered mean-zero field.
    """
 
    def __init__(self, Lx: float, alpha: float, tau: float, scale: float, min_eig: float = 1e-12):
        super().__init__()
        self.Lx = float(Lx)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.min_eig = float(min_eig)
        self.scale = float(scale)
        # cache eigenvalues across resolutions/dtypes
        self._lam_cache = {}

    def _lam(self, Nx, device, dtype):
        key = (Nx, device, dtype)
        if key not in self._lam_cache:
            lam = neumann_eigs_1d(Nx, self.Lx, device=device, dtype=dtype)  # shape (Nx,)
            self._lam_cache[key] = lam
        return self._lam_cache[key]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Expected (B, C, Nx)"
        B, C, Nx = z.shape

        # 1D DCT of input
        Zc = dct1(z)                      # (B, C, Nx)

        # Eigenvalues
        lam = self._lam(Nx, z.device, z.dtype)   # (Nx,)

        # (λ + τ²)^{-α/2}
        lamA = torch.clamp(lam + self.tau**2, min=self.min_eig)
        mult = self.scale*lamA.pow(-0.5 * self.alpha)       # (Nx,)

        # Spectral filtering
        Uc = Zc * mult[None, None, :]            # broadcast over (B,C)

        # Remove zero mode
        Uc[..., 0] = 0.0

        # Inverse DCT
        u = idct1(Uc)

        return u


# ============================================
# Cosine FNO backbone (1D)
# ============================================

class CosineFNOblock1d(nn.Module):
    """
    One residual layer of a 1D Cosine-FNO (Neumann FNO) network.

    Operations:
        1) Spectral convolution in the 1D cosine basis
           using CosineSpectralConv1d (mesh-invariant).
        2) Pointwise 1×1 convolution (local mixing).
        3) GELU activation.
        4) Residual skip connection:  y ← GELU( spec(x) + x ).

    Mesh Invariance
    ---------------
    `modes_x` may be:
        - int (e.g., 32): keep first 32 cosine modes
        - float in (0,1] (e.g., 0.25): keep that fraction of low frequencies
    Fractional selection → resolution-invariant FNO.

    Input
    -----
    x : (B, width, Nx)

    Output
    ------
    y : (B, width, Nx)
    """

    def __init__(self, width: int, modes_x: float | int, dropout: float = 0.0):
        super().__init__()

        # Mesh-invariant spectral convolution (1D)
        self.spec = CosineSpectralConv1d(
            in_channels=width,
            out_channels=width,
            modes_x=modes_x,
        )

        # Local 1×1 pointwise convolution
        self.w_pt = nn.Conv1d(width, width, kernel_size=1)

        # Activation + optional dropout
        self.act = nn.GELU()
        self.do  = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Spectral convolution
        y = self.spec(x)          # (B, width, Nx)

        # 2) Local mixing
        y = self.w_pt(y)

        # 3) Residual + activation + dropout
        y = self.act(y + x)
        y = self.do(y)

        return y


class CosineFNO1d(nn.Module):
    """
    Practical 1D Cosine-FNO for Neumann BC.

    Pipeline:
        lift (1x1 Conv1d)
        -> N spectral residual blocks (CosineFNOblock1d)
        -> proj (1x1 Conv1d)
        -> optional C^{1/2} projection

    Args
    ----
    Cin, Cout : channels
    width     : hidden width
    modes_x   : spectral cutoff (int or float)
    Lx        : physical length of domain
    depth     : number of CosineFNOblock1d layers
    use_cm    : apply C^{1/2} at the end
    cm_alpha  : alpha in C = A^{-alpha}
    cm_tau    : tau in A = -Δ + τ^2 I
    dropout   : dropout rate
    """
  
    def __init__(
        self,
        Cin: int,
        Cout: int,
        width: int,
        modes_x: float | int,
        Lx: float,
        depth: int = 4,
        use_cm: bool = True,
        cm_alpha: float = 1.0,
        cm_tau: float = 1.0,
        cm_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Lifting layer (local channel expansion)
        self.lift = nn.Conv1d(Cin, width, kernel_size=1)

        # Stack of spectral residual blocks
        self.blocks = nn.ModuleList(
            [
                CosineFNOblock1d(width, modes_x, dropout=dropout)
                for _ in range(depth)
            ]
        )

        # Projection back to output channels
        self.proj = nn.Conv1d(width, Cout, kernel_size=1)

        # Optional Cameron–Martin projection
        self.cm = (
            CMProjectionNeumann1d(Lx, alpha=cm_alpha, tau=cm_tau, scale=cm_scale)
            if use_cm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, Nx)
        y: (B, Cout, Nx)
        """
        y = self.lift(x)               # (B, width, Nx)
        for blk in self.blocks:
            y = blk(y)                 # (B, width, Nx)
        y = self.proj(y)               # (B, Cout, Nx)
        y = self.cm(y)                 # (B, Cout, Nx)
        return y


# ============================================
# Spectral upsampling (Neumann-consistent) — 1D
# ============================================

def spectral_upsample_neumann_1d(x_low: torch.Tensor, Nx_hi: int) -> torch.Tensor:
    """
    Upsample (B, C, Nx_low) -> (B, C, Nx_hi) by:
        DCT, zero-pad low modes into high grid, then inverse DCT.

    This preserves Neumann boundary conditions because the cosine basis
    is the eigenbasis of the 1D Neumann Laplacian.
    """
    B, C, Nx_l = x_low.shape

    # 1) DCT-II of low-resolution field
    Xl = dct1(x_low)                     # (B, C, Nx_l)

    # 2) Allocate high-resolution spectrum
    Xh = torch.zeros(B, C, Nx_hi,
                     device=x_low.device, dtype=x_low.dtype)

    # 3) Copy low-frequency modes into high-resolution spectrum
    m = min(Nx_l, Nx_hi)
    Xh[:, :, :m] = Xl[:, :, :m]

    # 4) Inverse DCT to obtain high-resolution field
    return idct1(Xh)                     # (B, C, Nx_hi)


# ============================================

# ============================================
class MultiInputCosineFNO1d(nn.Module):
    """
    1D version of MultiInputCosineFNO.

    Expects:
        u: (B, 1, Nx_u)
        y: (B, 1, Nx_y)

    Returns:
        out: (B, 1, Nx_u)
    """

    def __init__(self, base_fno: nn.Module,
                 mean_u: float, std_u: float,
                 mean_y: float, std_y: float):
        super().__init__()
        self.fno = base_fno   # e.g., CosineFNO1d(Cin=2, Cout=1, ...)
        
        # register normalizing constants
        self.register_buffer("mean_u", torch.tensor(mean_u, dtype=torch.float32))
        self.register_buffer("std_u",  torch.tensor(std_u, dtype=torch.float32))
        self.register_buffer("mean_y", torch.tensor(mean_y, dtype=torch.float32))
        self.register_buffer("std_y",  torch.tensor(std_y, dtype=torch.float32))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # --- normalize ---
        u_raw = u
        u = (u - self.mean_u) / self.std_u
        y = (y - self.mean_y) / self.std_y

        # Basic shape checks
        assert u.dim() == y.dim() ==  3, "Use (B,C,Nx) tensors."
        assert u.shape[1] == y.shape[1] == 1, "Each input must be 1 channel."

        # Target grid
        Nx_u = u.shape[-1]

        # Spectral upsampling (Neumann-consistent, 1D)
        y_up = spectral_upsample_neumann_1d(y, Nx_u)   # (B,1,Nx_u)

        # Concatenate channels → 3-channel signal
        x = torch.cat([u, y_up], dim=1)          # (B,2,Nx_u)

        # FNO backbone + identity (residual)
        out = u_raw + self.fno(x)                      # (B,1,Nx_u)
        return out

class FNO2Adapter1d(nn.Module):
    """
    Adapter for 1D FNO pipelines.

    Maps flattened vectors (u, ydag) of shape:
        u_vec    : (N, Nx_u)
        ydag_vec : (N, Nx_y)

    Into:
        u_img    : (N, 1, Nx_u)
        ydag_img : (N, 1, Nx_y)

    Applies:
        out_img = fields(u_img, ydag_img)

    And returns:
        out_vec : (N, Nx_u)

    Parameters
    ----------
    base_fno  : nn.Module
        A CosineFNO1d model with Cin=2, Cout=1.
    mean_u, std_u : float
        Normalization stats for u.
    mean_y, std_y : float
        Normalization stats for ydag.
    """

    def __init__(self, base_fno, mean_u, std_u, mean_y, std_y):
        super().__init__()
        self.fields = MultiInputCosineFNO1d(base_fno, mean_u, std_u, mean_y, std_y)
        self._shapes_set = False

    def _maybe_set_shapes(self, u_vec, y_vec):
        if self._shapes_set:
            return

        self.Nx_u = u_vec.size(1)   # length of u
        self.Nx_y = y_vec.size(1)   # length of y, ydag

        self._shapes_set = True

    def forward(self, u_vec, ydag_vec):
        """
        u_vec    : (N, Nx_u)
        ydag_vec : (N, Nx_y)
        """
        self._maybe_set_shapes(u_vec, ydag_vec)

        N = u_vec.size(0)

        # Reshape into 1D fields: (N, 1, Nx)
        u_img    = u_vec.reshape(N, 1, self.Nx_u)
        ydag_img = ydag_vec.reshape(N, 1, self.Nx_y)

        # Multi-input FNO
        out_img = self.fields(u_img, ydag_img)   # (N,1,Nx_u)

        # Return flattened vector
        return out_img.reshape(N, self.Nx_u)