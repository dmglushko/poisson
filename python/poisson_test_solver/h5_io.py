"""Deterministic HDF5 and JSON sidecar I/O compatible with C++ poisson_io."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from .validation import (
    MESH_MAGNITUDE_MAX,
    MESH_MAGNITUDE_MIN,
    PROBLEM_MAGNITUDE_MAX,
    PROBLEM_MAGNITUDE_MIN,
    magnitude_range_error,
)

SCHEMA_VERSION: int = 1
NONE_U32: int = 0xFFFFFFFF

BC_DIRICHLET: int = 0
BC_NEUMANN: int = 1
BC_ROBIN: int = 2


@dataclass(frozen=True)
class MeshData:
    nodes: np.ndarray                     # (N, 3) float64
    elements: np.ndarray                  # (E, 4) uint32
    element_materials: np.ndarray         # (E,) uint16
    boundary_elements: np.ndarray         # (B, 3) uint32
    boundary_element_bcs: np.ndarray      # (B,) uint16
    bc_type: np.ndarray                   # (C,) uint8 (0=Dirichlet, 1=Neumann, 2=Robin)
    bc_value: np.ndarray                  # (C,) float64
    bc_has_alpha: np.ndarray              # (C,) bool
    bc_alpha: np.ndarray                  # (C,) float64
    material_names: list[str]
    material_eps_r: np.ndarray            # (M,) float64
    material_screening_length: np.ndarray # (M,) float64
    material_is_metal: np.ndarray         # (M,) bool
    material_has_voltage: np.ndarray      # (M,) bool
    material_voltage: np.ndarray          # (M,) float64
    charge_positions: np.ndarray          # (Q, 3) float64
    charge_values: np.ndarray             # (Q,) int32
    neighbours: np.ndarray                # (E, 4) uint32
    screened: bool = True


@dataclass(frozen=True)
class AssembledProblemData:
    A: sparse.csc_matrix                  # (M, M) float64, sorted CSC with int32 indices
    f: np.ndarray                         # (M,) float64
    row: np.ndarray                       # (N,) uint32, NONE_U32 for known nodes
    known_node: np.ndarray                # (K,) uint32
    known_value: np.ndarray               # (K,) float64


@dataclass(frozen=True)
class AssemblyInfo:
    time: float
    time_assemble_A: float
    time_assemble_f: float
    nnz: int
    n_dirichlet: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SolverResultData:
    x: np.ndarray                         # (M,) float64


@dataclass(frozen=True)
class SolverInfo:
    time: float
    iterations: int
    residual: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NodalFieldData:
    u: np.ndarray                         # (N,) float64
    point_source_position: np.ndarray     # (Q, 3) float64
    point_source_value: np.ndarray        # (Q,) float64
    point_source_element: np.ndarray      # (Q,) uint32, NONE_U32 if unknown


@dataclass(frozen=True)
class FieldInfo:
    time: float
    n_nodes: int
    n_point_sources: int
    warnings: list[str] = field(default_factory=list)


def _canonicalize_f64(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype="<f8")
    if out.size == 0:
        return out
    return np.where(out == 0.0, 0.0, out).astype("<f8", copy=False)


def _canonicalize_scalar(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return 0.0 if value == 0.0 else float(value)


def _decode_str_attr(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.rstrip(b"\x00").decode("utf-8")
    if isinstance(raw, np.bytes_):
        return bytes(raw).rstrip(b"\x00").decode("utf-8")
    return str(raw)


def _attr_id(obj: h5py.Group | h5py.File, path: Path, name: str) -> Any:
    if name not in obj.attrs:
        raise RuntimeError(f"{path}: missing attribute '{name}'")
    attr = obj.attrs.get_id(name)
    if attr.shape != ():
        raise RuntimeError(f"{path}: attribute '{name}' must be a scalar")
    return attr


def _read_str_attr(obj: h5py.Group | h5py.File, path: Path, name: str) -> str:
    attr = _attr_id(obj, path, name)
    if h5py.check_string_dtype(attr.dtype) is None and attr.dtype.kind != "S":
        raise RuntimeError(f"{path}: attribute '{name}' is not a string")
    try:
        return _decode_str_attr(obj.attrs[name])
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{path}: attribute '{name}' is not valid UTF-8") from exc


_EXACT_UINT_TYPES: dict[str, Any] = {"<u4": h5py.h5t.STD_U32LE, "<u8": h5py.h5t.STD_U64LE}


def _read_uint_attr(obj: h5py.Group | h5py.File, path: Path, name: str, dtype: str) -> int:
    attr = _attr_id(obj, path, name)
    # Compare the on-disk HDF5 type itself (like C++ `H5Tequal`), not the numpy dtype h5py maps
    # it to: a 4-byte integer with 31-bit precision or a non-zero offset still maps to `<u4`.
    if not attr.get_type().equal(_EXACT_UINT_TYPES[dtype]):
        raise RuntimeError(
            f"{path}: attribute '{name}' must have type {np.dtype(dtype).name} ({dtype}), found {attr.dtype}"
        )
    return int(obj.attrs[name])


def _verify_schema(h5: h5py.File, path: Path, expected_schema: str) -> None:
    actual = _read_str_attr(h5, path, "schema")
    if actual != expected_schema:
        raise RuntimeError(f"{path}: expected schema '{expected_schema}', found '{actual}'")
    version = _read_uint_attr(h5, path, "schema_version", "<u4")
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"{path}: unsupported schema_version {version}, expected {SCHEMA_VERSION}")


def _open_group(parent: h5py.Group | h5py.File, path: Path, name: str) -> h5py.Group:
    obj = parent.get(name)
    if not isinstance(obj, h5py.Group):
        raise RuntimeError(f"{path}: missing group '{name}'")
    return obj


def _read_dataset(
    parent: h5py.Group | h5py.File,
    path: Path,
    name: str,
    dtype: str,
    cols: int | None = None,
) -> np.ndarray:
    """Read dataset `name` requiring the exact on-disk type and shape `(n,)` or `(n, cols)`.

    Mirrors C++ `read_dataset_1d` / `read_dataset_2d`: no implicit casts and no reshape, so a
    file with the wrong type or dimensions is rejected instead of silently reinterpreted.
    """
    full = f"{parent.name.rstrip('/')}/{name}".lstrip("/")
    ds = parent.get(name)
    if not isinstance(ds, h5py.Dataset):
        raise RuntimeError(f"{path}: missing dataset '{full}'")
    want = np.dtype(dtype)
    if ds.dtype != want:
        raise RuntimeError(f"{path}: dataset '{full}' must have type {want.name} ({dtype}), found {ds.dtype}")
    if ds.shape is None:
        raise RuntimeError(f"{path}: dataset '{full}' has a null dataspace")
    expected_ndim = 1 if cols is None else 2
    if len(ds.shape) != expected_ndim:
        raise RuntimeError(f"{path}: dataset '{full}' must have rank {expected_ndim}, found rank {len(ds.shape)}")
    if cols is not None and ds.shape[1] != cols:
        raise RuntimeError(f"{path}: dataset '{full}' must have {cols} columns, found {ds.shape[1]}")
    return np.array(ds[()], dtype=want, copy=True)


def _require_finite(path: Path, name: str, arr: np.ndarray) -> None:
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        bad = int(np.flatnonzero(~np.isfinite(arr.reshape(-1)))[0])
        raise RuntimeError(f"{path}: dataset '{name}' contains a non-finite value at flat index {bad}")


def _require_supported(path: Path, name: str, arr: np.ndarray, lo: float, hi: float) -> None:
    """Non-finite -> RuntimeError (bad file); nonzero |v| outside [lo, hi] -> ValueError."""
    _require_finite(path, name, arr)
    why = magnitude_range_error(arr, f"dataset '{name}'", lo, hi)
    if why is not None:
        raise ValueError(f"{path}: {why}")


def _require_mesh_range(path: Path, name: str, arr: np.ndarray) -> None:
    """Mesh / assembly inputs: nonzero magnitudes in [1e-30, 1e30]."""
    _require_supported(path, name, arr, MESH_MAGNITUDE_MIN, MESH_MAGNITUDE_MAX)


def _require_coordinate_range(path: Path, name: str, arr: np.ndarray) -> None:
    """Node / charge coordinates: |v| <= 1e30 (only their differences matter)."""
    _require_supported(path, name, arr, 0.0, MESH_MAGNITUDE_MAX)


def _require_problem_range(path: Path, name: str, arr: np.ndarray) -> None:
    """AssembledProblem values: nonzero magnitudes in [1e-100, 1e100]."""
    _require_supported(path, name, arr, PROBLEM_MAGNITUDE_MIN, PROBLEM_MAGNITUDE_MAX)


def _require_bool_flags(path: Path, name: str, arr: np.ndarray) -> np.ndarray:
    bad = np.flatnonzero(arr > 1)
    if bad.size > 0:
        raise RuntimeError(f"{path}: invalid boolean flag value {int(arr[bad[0]])} in '{name}'")
    return arr != 0


def _require_below(path: Path, name: str, arr: np.ndarray, limit: int, what: str) -> None:
    if arr.size > 0:
        top = int(np.max(arr))
        if top >= limit:
            raise RuntimeError(f"{path}: {name} references out-of-range {what} index {top}")


def _require_same_length(path: Path, what: str, *arrays: np.ndarray) -> None:
    if len({int(a.shape[0]) for a in arrays}) > 1:
        raise RuntimeError(f"{path}: {what} dataset size mismatch")


def _write_str_attr(obj: h5py.Group | h5py.File, name: str, value: str) -> None:
    encoded = value.encode("utf-8")
    size = max(1, len(encoded))
    obj.attrs.create(name, encoded, dtype=f"S{size}")


def _create_dataset(group: h5py.Group | h5py.File, name: str, data: np.ndarray, dtype: str) -> None:
    arr = np.ascontiguousarray(data, dtype=dtype)
    group.create_dataset(name, data=arr, dtype=dtype, track_times=False)


def _create_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    return parent.create_group(name, track_order=False)


def _open_deterministic_h5(path: Path) -> h5py.File:
    # Match C++ h5_common.cpp: HDF5 v1.8 format bounds, locking off for cluster filesystems.
    return h5py.File(path, "w", libver=("v108", "v108"), locking=False)


def _open_readonly_h5(path: Path) -> h5py.File:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: file not found")
    return h5py.File(path, "r", locking=False)


def _atomic_write(target: Path, writer: Any) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_mesh_properties(path: Path | str) -> MeshData:
    """Read and strictly validate a MeshProperties file (mirrors C++ `src/io/mesh_io.cpp`)."""
    p = Path(path)
    with _open_readonly_h5(p) as h5:
        _verify_schema(h5, p, "MeshProperties")
        screened_raw = _read_uint_attr(h5, p, "screened", "<u4")
        if screened_raw not in (0, 1):
            raise RuntimeError(f"{p}: attribute 'screened' must be 0 or 1")
        nodes = _read_dataset(h5, p, "nodes", "<f8", 3)
        _require_coordinate_range(p, "nodes", nodes)
        n_nodes = int(nodes.shape[0])

        g_bc = _open_group(h5, p, "boundary_conditions")
        bc_type = _read_dataset(g_bc, p, "type", "<u1")
        bc_value = _read_dataset(g_bc, p, "value", "<f8")
        bc_has_alpha_raw = _read_dataset(g_bc, p, "has_alpha", "<u1")
        bc_alpha = _read_dataset(g_bc, p, "alpha", "<f8")
        _require_same_length(p, "boundary_conditions", bc_type, bc_value, bc_has_alpha_raw, bc_alpha)
        if bc_type.size > 0 and int(np.max(bc_type)) > BC_ROBIN:
            raise RuntimeError(f"{p}: invalid BcType value {int(np.max(bc_type))}")
        bc_has_alpha = _require_bool_flags(p, "boundary_conditions/has_alpha", bc_has_alpha_raw)
        _require_mesh_range(p, "boundary_conditions/value", bc_value)
        _require_mesh_range(p, "boundary_conditions/alpha", bc_alpha)

        g_mat = _open_group(h5, p, "materials")
        name_bytes = _read_dataset(g_mat, p, "name_bytes", "<u1")
        name_offsets = _read_dataset(g_mat, p, "name_offsets", "<u8")
        material_eps_r = _read_dataset(g_mat, p, "eps_r", "<f8")
        material_screening_length = _read_dataset(g_mat, p, "screening_length", "<f8")
        is_metal_raw = _read_dataset(g_mat, p, "is_metal", "<u1")
        has_voltage_raw = _read_dataset(g_mat, p, "has_voltage", "<u1")
        material_voltage = _read_dataset(g_mat, p, "voltage", "<f8")
        n_mat = int(material_eps_r.shape[0])
        if name_offsets.shape[0] != n_mat + 1:
            raise RuntimeError(f"{p}: materials dataset size mismatch")
        _require_same_length(
            p, "materials", material_eps_r, material_screening_length, is_metal_raw, has_voltage_raw, material_voltage
        )
        if int(name_offsets[0]) != 0 or int(name_offsets[-1]) != name_bytes.shape[0]:
            raise RuntimeError(f"{p}: materials/name_offsets must span materials/name_bytes from 0 to size")
        if np.any(name_offsets[:-1] > name_offsets[1:]):   # unsigned compare, never wraps
            raise RuntimeError(f"{p}: invalid materials/name_offsets")
        raw_names = name_bytes.tobytes()
        try:
            material_names = [
                raw_names[int(name_offsets[i]) : int(name_offsets[i + 1])].decode("utf-8") for i in range(n_mat)
            ]
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{p}: materials/name_bytes is not valid UTF-8") from exc
        material_is_metal = _require_bool_flags(p, "materials/is_metal", is_metal_raw)
        material_has_voltage = _require_bool_flags(p, "materials/has_voltage", has_voltage_raw)
        _require_mesh_range(p, "materials/eps_r", material_eps_r)
        _require_mesh_range(p, "materials/screening_length", material_screening_length)
        _require_mesh_range(p, "materials/voltage", material_voltage)

        g_el = _open_group(h5, p, "elements")
        elements = _read_dataset(g_el, p, "node", "<u4", 4)
        element_materials = _read_dataset(g_el, p, "material", "<u2")
        if elements.shape[0] != element_materials.shape[0]:
            raise RuntimeError(f"{p}: elements/node and elements/material size mismatch")
        _require_below(p, "elements/node", elements, n_nodes, "node")
        _require_below(p, "elements/material", element_materials, n_mat, "material")
        n_elem = int(elements.shape[0])

        g_be = _open_group(h5, p, "boundary_elements")
        boundary_elements = _read_dataset(g_be, p, "node", "<u4", 3)
        boundary_element_bcs = _read_dataset(g_be, p, "boundary_condition", "<u2")
        if boundary_elements.shape[0] != boundary_element_bcs.shape[0]:
            raise RuntimeError(
                f"{p}: boundary_elements/node and boundary_elements/boundary_condition size mismatch"
            )
        _require_below(p, "boundary_elements/node", boundary_elements, n_nodes, "node")
        _require_below(
            p, "boundary_elements/boundary_condition", boundary_element_bcs, int(bc_type.shape[0]), "boundary condition"
        )

        g_ch = _open_group(h5, p, "charges")
        charge_positions = _read_dataset(g_ch, p, "position", "<f8", 3)
        charge_values = _read_dataset(g_ch, p, "value", "<i4")
        if charge_positions.shape[0] != charge_values.shape[0]:
            raise RuntimeError(f"{p}: charges dataset size mismatch")
        _require_coordinate_range(p, "charges/position", charge_positions)

        neighbours = _read_dataset(h5, p, "neighbours", "<u4", 4)
        if neighbours.shape[0] != n_elem:
            raise RuntimeError(f"{p}: neighbours length must match elements count")
        real_nb = neighbours[neighbours != NONE_U32]
        _require_below(p, "neighbours", real_nb, n_elem, "element")

    return MeshData(
        nodes=nodes,
        elements=elements,
        element_materials=element_materials,
        boundary_elements=boundary_elements,
        boundary_element_bcs=boundary_element_bcs,
        bc_type=bc_type,
        bc_value=bc_value,
        bc_has_alpha=bc_has_alpha,
        bc_alpha=bc_alpha,
        material_names=material_names,
        material_eps_r=material_eps_r,
        material_screening_length=material_screening_length,
        material_is_metal=material_is_metal,
        material_has_voltage=material_has_voltage,
        material_voltage=material_voltage,
        charge_positions=charge_positions,
        charge_values=charge_values,
        neighbours=neighbours,
        screened=(screened_raw != 0),
    )


def write_mesh_properties(path: Path | str, mesh: MeshData) -> None:
    """Write MeshProperties HDF5 (useful for constructing synthetic test meshes)."""

    def _write(tmp: Path) -> None:
        with _open_deterministic_h5(tmp) as h5:
            _write_str_attr(h5, "schema", "MeshProperties")
            h5.attrs.create("schema_version", np.uint32(SCHEMA_VERSION), dtype="<u4")
            h5.attrs.create("screened", np.uint32(1 if mesh.screened else 0), dtype="<u4")
            _create_dataset(h5, "nodes", _canonicalize_f64(mesh.nodes.reshape(-1, 3)), "<f8")

            g_el = _create_group(h5, "elements")
            _create_dataset(g_el, "node", mesh.elements.reshape(-1, 4), "<u4")
            _create_dataset(g_el, "material", mesh.element_materials.reshape(-1), "<u2")

            g_be = _create_group(h5, "boundary_elements")
            _create_dataset(g_be, "node", mesh.boundary_elements.reshape(-1, 3), "<u4")
            _create_dataset(g_be, "boundary_condition", mesh.boundary_element_bcs.reshape(-1), "<u2")

            g_bc = _create_group(h5, "boundary_conditions")
            _create_dataset(g_bc, "type", mesh.bc_type.reshape(-1), "<u1")
            _create_dataset(g_bc, "value", _canonicalize_f64(mesh.bc_value.reshape(-1)), "<f8")
            _create_dataset(g_bc, "has_alpha", mesh.bc_has_alpha.astype(np.uint8).reshape(-1), "<u1")
            _create_dataset(g_bc, "alpha", _canonicalize_f64(mesh.bc_alpha.reshape(-1)), "<f8")

            g_mat = _create_group(h5, "materials")
            name_bytes = bytearray()
            offsets = [0]
            for name in mesh.material_names:
                name_bytes.extend(name.encode("utf-8"))
                offsets.append(len(name_bytes))
            _create_dataset(g_mat, "name_bytes", np.frombuffer(name_bytes, dtype=np.uint8), "<u1")
            _create_dataset(g_mat, "name_offsets", np.asarray(offsets, dtype=np.uint64), "<u8")
            _create_dataset(g_mat, "eps_r", _canonicalize_f64(mesh.material_eps_r.reshape(-1)), "<f8")
            _create_dataset(
                g_mat, "screening_length", _canonicalize_f64(mesh.material_screening_length.reshape(-1)), "<f8"
            )
            _create_dataset(g_mat, "is_metal", mesh.material_is_metal.astype(np.uint8).reshape(-1), "<u1")
            _create_dataset(g_mat, "has_voltage", mesh.material_has_voltage.astype(np.uint8).reshape(-1), "<u1")
            _create_dataset(g_mat, "voltage", _canonicalize_f64(mesh.material_voltage.reshape(-1)), "<f8")

            g_ch = _create_group(h5, "charges")
            _create_dataset(g_ch, "position", _canonicalize_f64(mesh.charge_positions.reshape(-1, 3)), "<f8")
            _create_dataset(g_ch, "value", mesh.charge_values.reshape(-1), "<i4")

            _create_dataset(h5, "neighbours", mesh.neighbours.reshape(-1, 4), "<u4")

    _atomic_write(Path(path), _write)


def normalize_csc(A: sparse.spmatrix) -> sparse.csc_matrix:
    """Return a canonical compressed CSC matrix with sorted strictly-increasing int32 indices."""
    mat = A.tocsc(copy=True)
    mat.sum_duplicates()
    mat.eliminate_zeros()
    mat.sort_indices()
    if mat.shape[0] > 0x7FFFFFFF or mat.shape[1] > 0x7FFFFFFF or mat.nnz > 0x7FFFFFFF:
        raise OverflowError(f"Matrix dimensions/nnz exceed int32 Eigen StorageIndex range: {mat.shape}, nnz={mat.nnz}")
    mat.indptr = mat.indptr.astype("<i4", copy=False)
    mat.indices = mat.indices.astype("<i4", copy=False)
    mat.data = _canonicalize_f64(mat.data)
    return mat


def write_assembled_problem(path: Path | str, problem: AssembledProblemData) -> None:
    mat = normalize_csc(problem.A)
    rows, cols = mat.shape
    f_arr = _canonicalize_f64(problem.f.reshape(-1))
    if f_arr.shape[0] != rows:
        raise ValueError(f"f length ({f_arr.shape[0]}) must match A.rows ({rows})")
    row_arr = np.asarray(problem.row, dtype="<u4").reshape(-1)
    known_node = np.asarray(problem.known_node, dtype="<u4").reshape(-1)
    known_value = _canonicalize_f64(problem.known_value.reshape(-1))
    if known_node.shape[0] != known_value.shape[0]:
        raise ValueError("known_node and known_value length mismatch")

    def _write(tmp: Path) -> None:
        with _open_deterministic_h5(tmp) as h5:
            _write_str_attr(h5, "schema", "AssembledProblem")
            h5.attrs.create("schema_version", np.uint32(SCHEMA_VERSION), dtype="<u4")

            g_a = _create_group(h5, "A")
            _write_str_attr(g_a, "storage_order", "col_major")
            g_a.attrs.create("rows", np.uint64(rows), dtype="<u8")
            g_a.attrs.create("cols", np.uint64(cols), dtype="<u8")
            _create_dataset(g_a, "outer_index", mat.indptr, "<i4")
            _create_dataset(g_a, "inner_index", mat.indices, "<i4")
            _create_dataset(g_a, "values", mat.data, "<f8")

            _create_dataset(h5, "f", f_arr, "<f8")
            _create_dataset(h5, "row", row_arr, "<u4")

            g_known = _create_group(h5, "known")
            _create_dataset(g_known, "node", known_node, "<u4")
            _create_dataset(g_known, "value", known_value, "<f8")

    _atomic_write(Path(path), _write)


def read_assembled_problem(path: Path | str) -> AssembledProblemData:
    """Read and strictly validate an AssembledProblem file (mirrors `assembled_problem_io.cpp`)."""
    p = Path(path)
    with _open_readonly_h5(p) as h5:
        _verify_schema(h5, p, "AssembledProblem")
        g_a = _open_group(h5, p, "A")
        order = _read_str_attr(g_a, p, "storage_order")
        if order != "col_major":
            raise RuntimeError(f"{p}: unsupported A/storage_order '{order}', expected 'col_major'")
        rows = _read_uint_attr(g_a, p, "rows", "<u8")
        cols = _read_uint_attr(g_a, p, "cols", "<u8")
        if rows > 0x7FFFFFFF or cols > 0x7FFFFFFF:
            raise RuntimeError(f"{p}: A dimensions exceed Eigen StorageIndex range")
        indptr = _read_dataset(g_a, p, "outer_index", "<i4")
        indices = _read_dataset(g_a, p, "inner_index", "<i4")
        values = _read_dataset(g_a, p, "values", "<f8")
        if indptr.shape != (cols + 1,):
            raise RuntimeError(f"{p}: A/outer_index length must equal cols + 1")
        if indices.shape != values.shape:
            raise RuntimeError(f"{p}: A/inner_index and A/values length mismatch")
        nnz = int(indices.shape[0])
        if int(indptr[0]) != 0 or int(indptr[-1]) != nnz:
            raise RuntimeError(f"{p}: A/outer_index must start at 0 and end at nnz")
        if np.any(np.diff(indptr.astype(np.int64)) < 0):
            raise RuntimeError(f"{p}: A/outer_index is not non-decreasing")
        if nnz > 0:
            if int(np.min(indices)) < 0 or int(np.max(indices)) >= rows:
                bad = indices[(indices < 0) | (indices >= rows)][0]
                raise RuntimeError(f"{p}: A/inner_index references out-of-range row {int(bad)}")
            # Strictly increasing within each column: every step inside a column must be > 0.
            col_start = np.zeros(nnz, dtype=bool)
            col_start[indptr[:-1][np.diff(indptr) > 0]] = True
            steps = np.diff(indices.astype(np.int64))
            if np.any((steps <= 0) & ~col_start[1:]):
                raise RuntimeError(f"{p}: A/inner_index must be strictly increasing within each column")
        _require_problem_range(p, "A/values", values)
        A = sparse.csc_matrix((values, indices, indptr), shape=(rows, cols))

        f = _read_dataset(h5, p, "f", "<f8")
        if f.shape[0] != rows:
            raise RuntimeError(f"{p}: A.rows() does not match f.size()")
        _require_problem_range(p, "f", f)

        row = _read_dataset(h5, p, "row", "<u4")
        _require_below(p, "row", row[row != NONE_U32], int(f.shape[0]), "row")

        g_known = _open_group(h5, p, "known")
        known_node = _read_dataset(g_known, p, "node", "<u4")
        known_value = _read_dataset(g_known, p, "value", "<f8")
        if known_node.shape != known_value.shape:
            raise RuntimeError(f"{p}: known dataset size mismatch")
        if row.shape[0] > 0:
            _require_below(p, "known/node", known_node, int(row.shape[0]), "node")
        _require_problem_range(p, "known/value", known_value)
    return AssembledProblemData(A=A, f=f, row=row, known_node=known_node, known_value=known_value)


def write_solver_result(path: Path | str, result: SolverResultData) -> None:
    x_arr = _canonicalize_f64(result.x.reshape(-1))

    def _write(tmp: Path) -> None:
        with _open_deterministic_h5(tmp) as h5:
            _write_str_attr(h5, "schema", "SolverResult")
            h5.attrs.create("schema_version", np.uint32(SCHEMA_VERSION), dtype="<u4")
            _create_dataset(h5, "x", x_arr, "<f8")

    _atomic_write(Path(path), _write)


def read_solver_result(path: Path | str) -> SolverResultData:
    p = Path(path)
    with _open_readonly_h5(p) as h5:
        _verify_schema(h5, p, "SolverResult")
        x = _read_dataset(h5, p, "x", "<f8")
        _require_finite(p, "x", x)   # finite only: the field stage copies x, no arithmetic
    return SolverResultData(x=x)


def write_nodal_field(path: Path | str, field_data: NodalFieldData) -> None:
    u_arr = _canonicalize_f64(field_data.u.reshape(-1))
    pos = _canonicalize_f64(field_data.point_source_position.reshape(-1, 3))
    val = _canonicalize_f64(field_data.point_source_value.reshape(-1))
    elem = np.asarray(field_data.point_source_element, dtype="<u4").reshape(-1)
    if not (pos.shape[0] == val.shape[0] == elem.shape[0]):
        raise ValueError("point_sources dataset size mismatch")

    def _write(tmp: Path) -> None:
        with _open_deterministic_h5(tmp) as h5:
            _write_str_attr(h5, "schema", "NodalField")
            h5.attrs.create("schema_version", np.uint32(SCHEMA_VERSION), dtype="<u4")
            _create_dataset(h5, "u", u_arr, "<f8")
            g = _create_group(h5, "point_sources")
            _create_dataset(g, "position", pos, "<f8")
            _create_dataset(g, "value", val, "<f8")
            _create_dataset(g, "element", elem, "<u4")

    _atomic_write(Path(path), _write)


def read_nodal_field(path: Path | str) -> NodalFieldData:
    p = Path(path)
    with _open_readonly_h5(p) as h5:
        _verify_schema(h5, p, "NodalField")
        u = _read_dataset(h5, p, "u", "<f8")
        _require_finite(p, "u", u)
        g = _open_group(h5, p, "point_sources")
        pos = _read_dataset(g, p, "position", "<f8", 3)
        val = _read_dataset(g, p, "value", "<f8")
        elem = _read_dataset(g, p, "element", "<u4")
        _require_same_length(p, "point_sources", pos, val, elem)
        _require_finite(p, "point_sources/position", pos)
        _require_finite(p, "point_sources/value", val)
    return NodalFieldData(
        u=u,
        point_source_position=pos,
        point_source_value=val,
        point_source_element=elem,
    )


def _write_json(path: Path | str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2) + "\n"

    def _write(tmp: Path) -> None:
        tmp.write_text(text, encoding="utf-8")

    _atomic_write(Path(path), _write)


def write_assembly_info_json(path: Path | str, info: AssemblyInfo) -> None:
    _write_json(
        path,
        {
            "type": "AssemblyResult::Info",
            "time": _canonicalize_scalar(info.time),
            "time_assemble_A": _canonicalize_scalar(info.time_assemble_A),
            "time_assemble_f": _canonicalize_scalar(info.time_assemble_f),
            "nnz": int(info.nnz),
            "n_dirichlet": int(info.n_dirichlet),
            "warnings": list(info.warnings),
        },
    )


def write_solver_info_json(path: Path | str, info: SolverInfo) -> None:
    _write_json(
        path,
        {
            "type": "SolverResult::Info",
            "time": _canonicalize_scalar(info.time),
            "iterations": int(info.iterations),
            "residual": _canonicalize_scalar(info.residual),
            "warnings": list(info.warnings),
        },
    )


def write_field_info_json(path: Path | str, info: FieldInfo) -> None:
    _write_json(
        path,
        {
            "type": "NodalFieldBuild::Info",
            "time": _canonicalize_scalar(info.time),
            "n_nodes": int(info.n_nodes),
            "n_point_sources": int(info.n_point_sources),
            "warnings": list(info.warnings),
        },
    )
