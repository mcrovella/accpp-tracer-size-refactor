"""Omega (QK^T) SVD decomposition for attention heads."""

import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from .models import ModelConfig
from ._typecheck import typechecked


# Disk-cache format version. Bump when the on-disk layout / dataset names change.
CACHE_VERSION = 1


@typechecked
def get_omega_decomposition(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> tuple[
    Float[Tensor, "n_layers n_heads d_model d_head"],
    Float[Tensor, "n_layers n_heads d_head"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute SVD decomposition of Q@K^T (Omega) for all attention heads.

    Factorizes the attention weight matrix Omega = W_Q @ W_K^T into
    U @ diag(S) @ VT for each attention head, enabling decomposition of
    attention patterns into rank-1 components (singular vectors).

    The SVD is always computed in fp32 regardless of the model's dtype:
    ACC++ is numerically sensitive (same reason TF32 is disabled in
    ``Tracer.__init__``), and bf16/fp16 SVDs lose precision in the singular
    values. Returned tensors are fp32.

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    Returns:
        Tuple of (U, S, VT) where:
            U: Left singular vectors, shape (n_layers, n_heads, d_model, d_head).
            S: Singular values, shape (n_layers, n_heads, d_head).
            VT: Right singular vectors, shape (n_layers, n_heads, d_head, d_model).
    """
    rank = model.cfg.d_head

    # Cast to fp32 before SVD; output tensors are always fp32 (was: model dtype).
    W_Q = model.W_Q.float()
    W_K = model.W_K.float()

    omega = einsum(
        W_Q if not config.use_numpy_svd else W_Q.cpu(),
        W_K if not config.use_numpy_svd else W_K.cpu(),
        "n_layers n_heads d_model d_head, n_layers n_heads d_model_out d_head "
        "-> n_layers n_heads d_model d_model_out",
    )

    if config.use_numpy_svd:
        U_np, S_np, VT_np = np.linalg.svd(omega)
        U = torch.from_numpy(U_np[:, :, :, :rank]).to(device)
        S = torch.from_numpy(S_np[:, :, :rank]).to(device)
        VT = torch.from_numpy(VT_np[:, :, :rank, :]).to(device)
    else:
        U, S, VT = torch.linalg.svd(omega)
        U = U[:, :, :, :rank].to(device)
        S = S[:, :, :rank].to(device)
        VT = VT[:, :, :rank, :].to(device)

    return U, S, VT


@typechecked
def compute_weight_pseudoinverses(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> tuple[
    Float[Tensor, "n_layers n_heads d_head d_model"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute pseudoinverses of W_Q and W_K weight matrices.

    Used for computing bias offsets in the trace_firing algorithm.
    Uses numpy for models that require it for numerical stability.

    The pseudoinverse is always computed in fp32 regardless of the model's
    dtype (same rationale as ``get_omega_decomposition``). Returned tensors
    are fp32.

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    Returns:
        Tuple of (W_Q_pinv, W_K_pinv).
    """
    # Cast to fp32 before pinv; output tensors are always fp32 (was: model dtype).
    W_Q = model.W_Q.float()
    W_K = model.W_K.float()

    if config.use_numpy_svd:
        W_Q_pinv = torch.from_numpy(np.linalg.pinv(W_Q.cpu())).to(device)
        W_K_pinv = torch.from_numpy(np.linalg.pinv(W_K.cpu())).to(device)
    else:
        W_Q_pinv = torch.linalg.pinv(W_Q).to(device)
        W_K_pinv = torch.linalg.pinv(W_K).to(device)

    return W_Q_pinv, W_K_pinv


def _cache_filename(model_name: str, use_numpy_svd: bool) -> str:
    """Build the cache filename for a (model_name, use_numpy_svd) pair.

    Slashes in HF-style model names (e.g. ``EleutherAI/pythia-160m``) are
    replaced with ``__`` for filesystem-safety. The SVD backend
    (``torch`` vs ``numpy``) is suffixed because the two produce numerically
    distinct results.
    """
    safe_name = model_name.replace("/", "__").replace("\\", "__")
    suffix = "numpy" if use_numpy_svd else "torch"
    return f"{safe_name}_{suffix}.h5"


def _expected_shapes(model: HookedTransformer) -> dict[str, tuple[int, ...]]:
    """Expected (n_layers, n_heads, ...) shapes for the five cached tensors."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    return {
        "U": (n_layers, n_heads, d_model, d_head),
        "S": (n_layers, n_heads, d_head),
        "VT": (n_layers, n_heads, d_head, d_model),
        "W_Q_pinv": (n_layers, n_heads, d_head, d_model),
        "W_K_pinv": (n_layers, n_heads, d_head, d_model),
    }


def load_decomposition_cache(
    cache_dir: str | Path,
    model: HookedTransformer,
    use_numpy_svd: bool,
    device: str = "cpu",
) -> dict[str, Tensor] | None:
    """Try to load cached ``U, S, VT, W_Q_pinv, W_K_pinv`` from disk.

    Returns ``None`` (cache-miss) on any failure: missing file, unreadable
    file, missing datasets, or shape mismatch with the current model.
    The caller is expected to fall back to recomputation and re-save.

    The cache file uses h5py with gzip compression. All tensors are stored
    in fp32 and loaded back as fp32 onto ``device``.

    Args:
        cache_dir: Directory containing cache files.
        model: HookedTransformer model — used to validate cached shapes and
            to derive the cache filename from ``model.cfg.model_name``.
        use_numpy_svd: Selects the cache file variant (``torch`` vs
            ``numpy`` SVD backend).
        device: Torch device to place loaded tensors on.

    Returns:
        Dict with keys ``"U", "S", "VT", "W_Q_pinv", "W_K_pinv"`` mapping to
        fp32 tensors on ``device``, or ``None`` on cache-miss.
    """
    cache_path = Path(cache_dir) / _cache_filename(model.cfg.model_name, use_numpy_svd)
    if not cache_path.exists():
        return None

    expected = _expected_shapes(model)
    try:
        with h5py.File(cache_path, "r") as f:
            out: dict[str, Tensor] = {}
            for key, exp_shape in expected.items():
                if key not in f:
                    warnings.warn(
                        f"Cache file {cache_path} missing dataset '{key}'; "
                        "falling back to recomputation."
                    )
                    return None
                arr = f[key][:]
                if tuple(arr.shape) != exp_shape:
                    warnings.warn(
                        f"Cache file {cache_path} has shape {arr.shape} for '{key}' "
                        f"but model expects {exp_shape}; falling back to recomputation."
                    )
                    return None
                out[key] = torch.from_numpy(arr).to(device)
            return out
    except (OSError, KeyError, ValueError) as e:
        warnings.warn(
            f"Failed to load cache file {cache_path} ({type(e).__name__}: {e}); "
            "falling back to recomputation."
        )
        return None


def save_decomposition_cache(
    cache_dir: str | Path,
    U: Tensor,
    S: Tensor,
    VT: Tensor,
    W_Q_pinv: Tensor,
    W_K_pinv: Tensor,
    model_name: str,
    use_numpy_svd: bool,
    compression_level: int = 9,
) -> Path:
    """Save the decomposition tensors to a gzip-compressed h5 file.

    Tensors are written in fp32 to a file under ``cache_dir``. The directory
    is created if it does not exist.

    Args:
        cache_dir: Directory to write the cache file into.
        U, S, VT: Omega SVD tensors.
        W_Q_pinv, W_K_pinv: Weight pseudoinverses.
        model_name: ``model.cfg.model_name``; used to compose the filename.
        use_numpy_svd: Whether the SVD was computed with numpy (selects the
            ``_numpy.h5`` vs ``_torch.h5`` filename suffix).
        compression_level: gzip level 0–9 (default 9 — see CHANGELOG for
            rationale: this cache is written once and read many times, so
            the asymmetric workload makes the slower write irrelevant).

    Returns:
        The path of the written file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_filename(model_name, use_numpy_svd)

    tensors = {
        "U": U,
        "S": S,
        "VT": VT,
        "W_Q_pinv": W_Q_pinv,
        "W_K_pinv": W_K_pinv,
    }
    with h5py.File(cache_path, "w") as f:
        for key, tensor in tensors.items():
            f.create_dataset(
                key,
                data=tensor.detach().cpu().float().numpy(),
                compression="gzip",
                compression_opts=compression_level,
            )
        f.attrs["model_name"] = model_name
        f.attrs["use_numpy_svd"] = use_numpy_svd
        f.attrs["cache_version"] = CACHE_VERSION

    return cache_path
