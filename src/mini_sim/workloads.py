"""Generate full synthetic workloads and runnable tiles for the dense baseline.

Adapted from the user's generate_benchmarks.py. This is data preparation;
it does not change the array or introduce sparse execution.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from base.dtype import cast_vector, normalize_dtype
from .benchmark import Config
from .reference import reference_product
from .reference import sparse_reference_product
from .sparse_codec import compress_2of4


# Preserve the user's dimensions and ordering: M, N, K.
WORKLOADS = {
    "resnet50_conv1": (12544, 64, 160),
    "llama3_q_proj": (32, 1024, 4096),
    "llama3_gate_proj": (256, 1024, 4096),
}


def sparsity_2_4(dense_weights, axis="K", *, return_mask=False):
    """Keep the two largest magnitudes in each four-element group.

    W has shape [K,N]. K groups four consecutive rows per output column;
    N reproduces the original script's grouping across columns. Stable sorting
    resolves equal magnitudes by preferring the earlier position. A retained
    entry can itself be zero; the mask always retains exactly two positions.
    """
    w = np.asarray(dense_weights)
    if w.ndim != 2 or not np.issubdtype(w.dtype, np.floating):
        raise ValueError("weights must be a floating-point [K,N] matrix")
    if axis not in ("K", "N"):
        raise ValueError("sparsity axis must be K or N")
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    # Work on rows whose last axis is the chosen grouping dimension.
    oriented = w.T if axis == "K" else w
    if oriented.shape[1] % 4:
        raise ValueError(f"{axis} dimension must be divisible by four")
    groups = oriented.reshape(oriented.shape[0], -1, 4)
    keep = np.argsort(-np.abs(groups.astype(np.float32)), axis=-1, kind="stable")[..., :2]
    mask = np.zeros(groups.shape, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    mask = mask.reshape(oriented.shape)
    if axis == "K":
        mask = mask.T
    sparse = np.where(mask, w, np.zeros((), dtype=w.dtype))
    return (sparse, mask) if return_mask else sparse


def extract_tile(a, w, *, size=16, rows=64, row_start=0, k_start=0, n_start=0):
    """Select an in-bounds tile; pad only the final K/N edges with zeros.

    Returned A[rows,size] @ W[size,size] is one K-slice contribution. It is
    not a complete layer output when the original K exceeds size.
    """
    a, w = np.asarray(a), np.asarray(w)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0]:
        raise ValueError("expected compatible A[M,K] and W[K,N]")
    Config(size=size, rows=rows)
    if not (0 <= row_start < a.shape[0] and 0 <= k_start < a.shape[1]
            and 0 <= n_start < w.shape[1]):
        raise ValueError("tile starting offsets must be inside the workload")
    actual_rows = min(rows, a.shape[0] - row_start)
    valid_k = min(size, a.shape[1] - k_start)
    valid_n = min(size, w.shape[1] - n_start)
    at = np.zeros((actual_rows, size), dtype=a.dtype)
    wt = np.zeros((size, size), dtype=w.dtype)
    at[:, :valid_k] = a[row_start:row_start + actual_rows, k_start:k_start + valid_k]
    wt[:valid_k, :valid_n] = w[k_start:k_start + valid_k, n_start:n_start + valid_n]
    return at, wt, dict(row_start=row_start, k_start=k_start, n_start=n_start,
                       rows=actual_rows, size=size, valid_k=valid_k, valid_n=valid_n)


def _quantize(values, dtype):
    storage = np.float16 if dtype == "fp16" else np.float32
    if dtype == "fp32":
        return np.asarray(values, dtype=storage)
    dt = normalize_dtype(dtype)
    return np.asarray([cast_vector(row, dt) for row in values], dtype=storage)


def generate_benchmark(output_dir="benchmarks", *, seed=100, dtype="fp16",
                       names=None, tile_size=16, tile_rows=64, row_start=0,
                       k_start=0, n_start=0, sparsity_axis="K", full_reference=False,
                       workloads=None):
    """Write a manifest, full inputs, and one dense/pruned tile pair per workload.

    Full FP32 NumPy reference matmuls are opt-in diagnostics. Tile references
    model the original binary adder tree in the selected numeric format.
    """
    definitions = WORKLOADS if workloads is None else workloads
    selected = list(definitions) if names is None else list(names)
    if not selected or len(set(selected)) != len(selected) or any(name not in definitions for name in selected):
        raise ValueError("select unique known workload names")
    Config(size=tile_size, rows=tile_rows, dtype=dtype)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if sparsity_axis not in ("K", "N"):
        raise ValueError("sparsity axis must be K or N")
    if tile_size % 4 or k_start % 4 or n_start % 4:
        raise ValueError("tile size and K/N offsets must be multiples of four to preserve groups")
    # Validate all requests before writing any files.
    for name in selected:
        m, n, k = definitions[name]
        if min(m, n, k) < 1 or (k if sparsity_axis == "K" else n) % 4:
            raise ValueError(f"invalid workload dimensions for {name}")
        if not (0 <= row_start < m and 0 <= k_start < k and 0 <= n_start < n):
            raise ValueError(f"tile starting offsets outside {name}")
    root = Path(output_dir)
    (root / "full").mkdir(parents=True, exist_ok=True)
    (root / "tiles").mkdir(exist_ok=True)
    manifest = dict(schema_version=1, generator="numpy_pcg64", seed=seed, dtype=dtype,
                    synthetic=True, sparsity_axis=sparsity_axis,
                    hardware="original_dense_meissa", workloads=[])
    for name in selected:
        m, n, k = definitions[name]
        # Seed each named workload independently, so filtering doesn't change it.
        ordinal = list(definitions).index(name)
        rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
        a = _quantize(rng.standard_normal((m, k), dtype=np.float32), dtype)
        dense = _quantize(rng.standard_normal((k, n), dtype=np.float32), dtype)
        sparse, mask = sparsity_2_4(dense, sparsity_axis, return_mask=True)
        entry = dict(name=name, shape_MNK=[m, n, k], files={}, commands=[])
        for label, w in (("dense", dense), ("sparse", sparse)):
            meta = dict(workload=name, shape_MNK=[m, n, k], dtype=dtype,
                        sparsity_axis=sparsity_axis if label == "sparse" else None,
                        variant=label, synthetic=True, seed=seed, workload_seed_index=ordinal)
            full_path = Path("full") / f"{name}_{label}.npz"
            arrays = dict(A=a, W=w, metadata_json=np.asarray(json.dumps(meta)))
            if label == "sparse":
                arrays["keep_mask"] = mask
                if sparsity_axis == "K":
                    arrays["W_values"], arrays["W_indices"] = compress_2of4(w)
            np.savez(root / full_path, **arrays)
            at, wt, tile_info = extract_tile(a, w, size=tile_size, rows=tile_rows,
                row_start=row_start, k_start=k_start, n_start=n_start)
            tile_meta = dict(meta, tile=tile_info, reference_tree="binary",
                             scope="single_K_slice_contribution")
            reference_dtype = None if dtype == "fp32" else dtype
            reference = reference_product(at, wt, dtype=reference_dtype, mixed=False)
            tile_path = Path("tiles") / f"{name}_{label}.npz"
            np.savez(root / tile_path, A=at, W=wt, C_reference=reference,
                     metadata_json=np.asarray(json.dumps(tile_meta)))
            entry["files"][label] = dict(full=full_path.as_posix(), tile=tile_path.as_posix())
            entry["tile"] = tile_info
            # Double quotes work in the documented common shells, including paths with spaces.
            path_text = (root / tile_path).as_posix()
            entry["commands"].append(f'mini-sim --size {tile_size} --rows {at.shape[0]} '
                f'--dtype {dtype} --input "{path_text}"')
        if sparsity_axis == "K":
            # Same physical SxS grid, logical K=2*S. Existing square dense-
            # execution fixtures stay available for backwards compatibility.
            at, wide_w, tile_info = extract_tile(a,sparse,size=2*tile_size,rows=tile_rows,
                row_start=row_start,k_start=k_start,n_start=n_start)
            wt = wide_w[:,:tile_size]
            v,ix = compress_2of4(wt)
            tile_info.update(size=tile_size,logical_k=2*tile_size,valid_n=min(tile_size,n-n_start))
            structured_path = Path("tiles") / f"{name}_structured.npz"
            structured_meta = dict(workload=name,shape_MNK=[m,n,k],dtype=dtype,
                sparsity_axis="K",execution_mode="sparse",physical_size=tile_size,
                encoding="column_major_values_then_packed_2bit_indices",
                reference_tree="binary",tile=tile_info,synthetic=True,
                scope="single_K_slice_contribution")
            reference = sparse_reference_product(at,v,ix,
                dtype=None if dtype=="fp32" else dtype,mixed=False)
            np.savez(root/structured_path,A=at,W_values=v,W_indices=ix,
                C_reference=reference,metadata_json=np.asarray(json.dumps(structured_meta)))
            entry["files"]["structured"] = dict(tile=structured_path.as_posix())
            entry["structured_tile"] = tile_info
            entry["commands"].append(f'mini-sim --input "{(root/structured_path).as_posix()}" --load-path atalla')
        if full_reference:
            ref_path = Path("full") / f"{name}_numpy_reference.npz"
            af = a.astype(np.float32)
            np.savez(root / ref_path,
                     C_dense_fp32=af @ dense.astype(np.float32),
                     C_sparse_fp32=af @ sparse.astype(np.float32))
            entry["numpy_reference"] = ref_path.as_posix()
            entry["numpy_reference_scope"] = "diagnostic_full_matmul_not_MEISSA_golden"
        manifest["workloads"].append(entry)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate SoCET workloads and runnable dense MEISSA tiles")
    p.add_argument("--output-dir", default="benchmarks")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--workload", dest="names", choices=list(WORKLOADS), action="append")
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--tile-rows", type=int, default=64)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--k-start", type=int, default=0)
    p.add_argument("--n-start", type=int, default=0)
    p.add_argument("--sparsity-axis", choices=["K", "N"], default="K")
    p.add_argument("--full-reference", action="store_true", help="also compute diagnostic full FP32 NumPy matmuls")
    try:
        manifest = generate_benchmark(**vars(p.parse_args(argv)))
    except (ValueError, OSError) as exc:
        p.exit(2, f"generate-benchmarks: {exc}\n")
    for workload in manifest["workloads"]:
        print(f'{workload["name"]}: M,N,K={workload["shape_MNK"]}; selected tile commands:')
        for command in workload["commands"]:
            print(command)
    return 0

