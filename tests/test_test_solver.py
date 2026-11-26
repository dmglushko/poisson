"""Unit and end-to-end tests for python/poisson_test_solver."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from poisson_runner import Cache, Case, RunContext, compare, dirichlet, neumann, robin, run_experiment  # noqa: E402
from poisson_runner.stages import resolve_binary  # noqa: E402

# numpy, scipy, and h5py are optional for the rest of the CTest `runner` suite (`unittest
# discover`), so a missing one skips these tests instead of failing the whole discovery.
try:
    import h5py
    import numpy as np
    from scipy import sparse

    from poisson_test_solver import (
        COULOMB_CONSTANT_NM,
        E_OVER_EPS0_NM,
        NONE_U32,
        AssembledProblemData,
        MeshData,
        NodalFieldData,
        SolverResultData,
        assemble_problem,
        build_nodal_field,
        canonicalize_assembled_problem,
        compare_assembled_problems,
        read_assembled_problem,
        read_mesh_properties,
        read_nodal_field,
        read_solver_result,
        solve_problem,
        write_assembled_problem,
        write_mesh_properties,
        write_nodal_field,
        write_solver_result,
    )
    from poisson_test_solver import cli as ts_cli
    from poisson_test_solver import stages as ts_stages
    from poisson_test_solver.h5_io import BC_DIRICHLET, BC_NEUMANN, BC_ROBIN
    from poisson_test_solver.locate import compute_face_neighbours, locate_charges
    from poisson_test_solver.stages import make_solve_stage
    from poisson_test_solver.stages import assemble as stage_assemble
    from poisson_test_solver.stages import build_field as stage_build_field
    from poisson_test_solver.stages import solve as stage_solve
    from poisson_test_solver.stages import solve_mesh

    HAS_CORE_DEPS = True
    CORE_DEPS_SKIP_REASON = ""
except ImportError as _exc:
    HAS_CORE_DEPS = False
    CORE_DEPS_SKIP_REASON = f"requires numpy, scipy, and h5py ({_exc})"
    BC_DIRICHLET, BC_NEUMANN, BC_ROBIN = 0, 1, 2   # referenced by default arguments below

HAS_SKFEM = importlib.util.find_spec("skfem") is not None
HAS_PYAMG = importlib.util.find_spec("pyamg") is not None
HAS_GMSH = shutil.which("gmsh") is not None


def _build_box_mesh(
    nx: int,
    ny: int,
    nz: int,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    material_fn=None,
    materials=None,
    left_bc: tuple[int, float, float | None] = (BC_DIRICHLET, 0.0, None),
    right_bc: tuple[int, float, float | None] = (BC_DIRICHLET, 1.0, None),
    side_bc: tuple[int, float, float | None] | None = None,
    charges: list[tuple[tuple[float, float, float], int]] | None = None,
    screened: bool = True,
) -> MeshData:
    """Create a structured 3D Kuhn-triangulated box mesh [0, lx] x [0, ly] x [0, lz]."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)

    def nid(i: int, j: int, k: int) -> int:
        return (i * (ny + 1) + j) * (nz + 1) + k

    nodes = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float64)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes[nid(i, j, k)] = (xs[i], ys[j], zs[k])

    # 6 positively-oriented tets per cube (Kuhn triangulation along diagonal 000 -> 111)
    kuhn = [
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 1), (0, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 1), (1, 0, 0), (1, 1, 1)),
    ]

    elems = []
    elem_mats = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = 0.5 * (xs[i] + xs[i + 1])
                cy = 0.5 * (ys[j] + ys[j + 1])
                cz = 0.5 * (zs[k] + zs[k + 1])
                m_id = 0 if material_fn is None else int(material_fn(cx, cy, cz))
                for tet in kuhn:
                    v = [nid(i + di, j + dj, k + dk) for di, dj, dk in tet]
                    # Ensure positive orientation
                    a, b, c, d = nodes[v[0]], nodes[v[1]], nodes[v[2]], nodes[v[3]]
                    if float(np.dot(np.cross(b - a, c - a), d - a)) < 0.0:
                        v[1], v[2] = v[2], v[1]
                    elems.append(v)
                    elem_mats.append(m_id)

    bcs_list = [left_bc, right_bc]
    if side_bc is not None:
        bcs_list.append(side_bc)

    be_nodes = []
    be_bcs = []
    # x = 0 (left, bc 0) and x = lx (right, bc 1)
    for j in range(ny):
        for k in range(nz):
            # Left face (0,j,k) - (0,j+1,k) - (0,j+1,k+1) - (0,j,k+1); matches Kuhn diagonal (0,0,0)->(0,1,1)
            be_nodes.append([nid(0, j, k), nid(0, j + 1, k), nid(0, j + 1, k + 1)])
            be_bcs.append(0)
            be_nodes.append([nid(0, j, k), nid(0, j + 1, k + 1), nid(0, j, k + 1)])
            be_bcs.append(0)
            # Right face: diagonal (1,0,0)->(1,1,1)
            be_nodes.append([nid(nx, j, k), nid(nx, j + 1, k), nid(nx, j + 1, k + 1)])
            be_bcs.append(1)
            be_nodes.append([nid(nx, j, k), nid(nx, j + 1, k + 1), nid(nx, j, k + 1)])
            be_bcs.append(1)

    if side_bc is not None:
        for i in range(nx):
            for k in range(nz):
                # y = 0: diagonal (0,0,0)->(1,0,1)
                be_nodes.append([nid(i, 0, k), nid(i + 1, 0, k), nid(i + 1, 0, k + 1)])
                be_bcs.append(2)
                be_nodes.append([nid(i, 0, k), nid(i + 1, 0, k + 1), nid(i, 0, k + 1)])
                be_bcs.append(2)
                # y = ly: diagonal (0,1,0)->(1,1,1)
                be_nodes.append([nid(i, ny, k), nid(i + 1, ny, k), nid(i + 1, ny, k + 1)])
                be_bcs.append(2)
                be_nodes.append([nid(i, ny, k), nid(i + 1, ny, k + 1), nid(i, ny, k + 1)])
                be_bcs.append(2)
        for i in range(nx):
            for j in range(ny):
                # z = 0: diagonal (0,0,0)->(1,1,0)
                be_nodes.append([nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0)])
                be_bcs.append(2)
                be_nodes.append([nid(i, j, 0), nid(i + 1, j + 1, 0), nid(i, j + 1, 0)])
                be_bcs.append(2)
                # z = lz: diagonal (0,0,1)->(1,1,1)
                be_nodes.append([nid(i, j, nz), nid(i + 1, j, nz), nid(i + 1, j + 1, nz)])
                be_bcs.append(2)
                be_nodes.append([nid(i, j, nz), nid(i + 1, j + 1, nz), nid(i, j + 1, nz)])
                be_bcs.append(2)

    if materials is None:
        materials = [("diel", 1.0, 0.0, False, None)]

    ch_pos = np.asarray([c[0] for c in charges], dtype=np.float64).reshape(-1, 3) if charges else np.zeros((0, 3))
    ch_val = np.asarray([c[1] for c in charges], dtype=np.int32).reshape(-1) if charges else np.zeros((0,), dtype=np.int32)

    elems_arr = np.asarray(elems, dtype=np.uint32)
    return MeshData(
        nodes=nodes,
        elements=elems_arr,
        element_materials=np.asarray(elem_mats, dtype=np.uint16),
        boundary_elements=np.asarray(be_nodes, dtype=np.uint32).reshape(-1, 3),
        boundary_element_bcs=np.asarray(be_bcs, dtype=np.uint16),
        bc_type=np.asarray([b[0] for b in bcs_list], dtype=np.uint8),
        bc_value=np.asarray([b[1] for b in bcs_list], dtype=np.float64),
        bc_has_alpha=np.asarray([b[2] is not None for b in bcs_list], dtype=bool),
        bc_alpha=np.asarray([0.0 if b[2] is None else b[2] for b in bcs_list], dtype=np.float64),
        material_names=[m[0] for m in materials],
        material_eps_r=np.asarray([m[1] for m in materials], dtype=np.float64),
        material_screening_length=np.asarray([m[2] for m in materials], dtype=np.float64),
        material_is_metal=np.asarray([m[3] for m in materials], dtype=bool),
        material_has_voltage=np.asarray([m[4] is not None for m in materials], dtype=bool),
        material_voltage=np.asarray([0.0 if m[4] is None else m[4] for m in materials], dtype=np.float64),
        charge_positions=ch_pos,
        charge_values=ch_val,
        neighbours=compute_face_neighbours(elems_arr),
        screened=screened,
    )


def _replace_mesh(mesh: MeshData, **changes) -> MeshData:
    fields = {name: getattr(mesh, name) for name in MeshData.__dataclass_fields__}
    fields.update(changes)
    return MeshData(**fields)


def _mutate_h5(src: Path, dst: Path, mutate) -> Path:
    """Copy `src` to `dst` and apply `mutate(h5py.File)` in place."""
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+", locking=False) as h5:
        mutate(h5)
    return dst


def _replace_dataset(h5: h5py.File, name: str, data: np.ndarray, dtype: str | None = None) -> None:
    del h5[name]
    h5.create_dataset(name, data=data, dtype=dtype if dtype is not None else data.dtype)


def _set_31bit_screened(h5: h5py.File) -> None:
    """Rewrite `screened` as a 4-byte little-endian unsigned integer with 31-bit precision: h5py
    still maps it to numpy `<u4`, but it is not `H5T_STD_U32LE`."""
    del h5.attrs["screened"]
    tid = h5py.h5t.STD_U32LE.copy()
    tid.set_precision(31)
    attr = h5py.h5a.create(h5.id, b"screened", tid, h5py.h5s.create(h5py.h5s.SCALAR))
    attr.write(np.array(1, dtype="<u4"))


def _wrap_name_offsets(h5: h5py.File) -> None:
    """Three materials whose uint64 name_offsets go backwards past 2**63 (wrap as int64)."""
    _replace_dataset(h5, "materials/name_bytes", np.frombuffer(b"abc", dtype=np.uint8), "<u1")
    _replace_dataset(h5, "materials/name_offsets", np.array([0, 2**63 - 1, 2**64 - 2, 3], dtype=np.uint64), "<u8")
    for name, value, dtype in (("eps_r", 1.0, "<f8"), ("screening_length", 0.0, "<f8"), ("voltage", 0.0, "<f8"),
                               ("is_metal", 0, "<u1"), ("has_voltage", 0, "<u1")):
        _replace_dataset(h5, f"materials/{name}", np.full(3, value, dtype=dtype), dtype)


def _run_cli(argv: list[str]) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = ts_cli.main(argv)
    return code, err.getvalue()


def _box_mesh_h5(tmp: Path, name: str = "box_mesh.h5", **kwargs) -> Path:
    path = tmp / name
    write_mesh_properties(path, _build_box_mesh(**{"nx": 4, "ny": 2, "nz": 2, "screened": False, **kwargs}))
    return path


@unittest.skipUnless(HAS_CORE_DEPS, CORE_DEPS_SKIP_REASON)
class TestIOAndCoreHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cpp_roundtrip_and_constants_parity(self) -> None:
        try:
            tool = resolve_binary("io_roundtrip_tool")
        except FileNotFoundError:
            self.skipTest("io_roundtrip_tool not built")

        # 1. Exact constant parity with include/common/constants.h
        out = subprocess.check_output([str(tool), "constants"], text=True).strip().split()
        cpp_e_eps0, cpp_coulomb = float(out[0]), float(out[1])
        self.assertAlmostEqual(E_OVER_EPS0_NM, cpp_e_eps0, places=14)
        self.assertAlmostEqual(COULOMB_CONSTANT_NM, cpp_coulomb, places=14)

        # 2. AssembledProblem Python -> C++ read/write -> Python read
        A = sparse.csc_matrix(np.array([[2.0, -1.0, 0.0], [-1.0, 2.5, 0.0], [0.0, 0.0, 4.0]]))
        ap = AssembledProblemData(
            A=A,
            f=np.array([1.0, -2.5, -0.0], dtype=np.float64),
            row=np.array([NONE_U32, 0, 1, 1, 2, NONE_U32], dtype=np.uint32),
            known_node=np.array([0, 5], dtype=np.uint32),
            known_value=np.array([-0.0, 5.0], dtype=np.float64),
        )
        py_ap = self.tmp / "py_ap.h5"
        cpp_ap = self.tmp / "cpp_ap.h5"
        write_assembled_problem(py_ap, ap)
        subprocess.run([str(tool), "assembled", str(py_ap), str(cpp_ap)], check=True)
        ap_rt = read_assembled_problem(cpp_ap)
        comp = compare_assembled_problems(ap, ap_rt)
        self.assertTrue(comp.equal, comp.reason)

        # 3. Empty AssembledProblem round-trip
        ap_empty = AssembledProblemData(
            A=sparse.csc_matrix((0, 0), dtype=np.float64),
            f=np.zeros((0,), dtype=np.float64),
            row=np.full((2,), NONE_U32, dtype=np.uint32),
            known_node=np.array([0, 1], dtype=np.uint32),
            known_value=np.array([1.0, 1.0], dtype=np.float64),
        )
        write_assembled_problem(self.tmp / "py_empty.h5", ap_empty)
        subprocess.run(
            [str(tool), "assembled", str(self.tmp / "py_empty.h5"), str(self.tmp / "cpp_empty.h5")],
            check=True,
        )
        ap_empty_rt = read_assembled_problem(self.tmp / "cpp_empty.h5")
        self.assertEqual(ap_empty_rt.A.shape, (0, 0))
        self.assertEqual(ap_empty_rt.f.size, 0)

        # 4. SolverResult Python -> C++ -> Python
        sr = SolverResultData(x=np.array([0.25, 1.5, -4.75], dtype=np.float64))
        write_solver_result(self.tmp / "py_sr.h5", sr)
        subprocess.run([str(tool), "solver", str(self.tmp / "py_sr.h5"), str(self.tmp / "cpp_sr.h5")], check=True)
        sr_rt = read_solver_result(self.tmp / "cpp_sr.h5")
        np.testing.assert_array_equal(sr_rt.x, sr.x)

        # 5. NodalField Python -> C++ -> Python
        nf = NodalFieldData(
            u=np.array([0.0, 1.25, -2.5, 3.0], dtype=np.float64),
            point_source_position=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
            point_source_value=np.array([-18.095, 36.19], dtype=np.float64),
            point_source_element=np.array([0, NONE_U32], dtype=np.uint32),
        )
        write_nodal_field(self.tmp / "py_nf.h5", nf)
        subprocess.run([str(tool), "nodal", str(self.tmp / "py_nf.h5"), str(self.tmp / "cpp_nf.h5")], check=True)
        nf_rt = read_nodal_field(self.tmp / "cpp_nf.h5")
        np.testing.assert_array_equal(nf_rt.u, nf.u)
        np.testing.assert_array_equal(nf_rt.point_source_position, nf.point_source_position)
        np.testing.assert_array_equal(nf_rt.point_source_value, nf.point_source_value)
        np.testing.assert_array_equal(nf_rt.point_source_element, nf.point_source_element)

    def test_hdf5_determinism_across_seconds(self) -> None:
        A = sparse.csc_matrix(np.array([[3.0, -1.0], [-1.0, 2.0]]))
        ap = AssembledProblemData(
            A=A,
            f=np.array([1.0, 2.0]),
            row=np.array([0, 1, NONE_U32], dtype=np.uint32),
            known_node=np.array([2], dtype=np.uint32),
            known_value=np.array([0.0]),
        )
        f1 = self.tmp / "det1.h5"
        f2 = self.tmp / "det2.h5"
        write_assembled_problem(f1, ap)
        time.sleep(1.1)
        write_assembled_problem(f2, ap)
        self.assertEqual(f1.read_bytes(), f2.read_bytes())

    def test_permutation_invariant_comparator(self) -> None:
        A = sparse.csc_matrix(np.array([[4.0, -1.0, 0.5], [-1.0, 5.0, -2.0], [0.5, -2.0, 6.0]]))
        f = np.array([1.0, 2.0, 3.0])
        # Nodes 2 & 4 share row 1 (floating metal); node 0 is known; node 1 -> row 0; node 3 -> row 2.
        p1 = AssembledProblemData(
            A=A,
            f=f,
            row=np.array([NONE_U32, 0, 1, 2, 1, NONE_U32], dtype=np.uint32),
            known_node=np.array([0, 5], dtype=np.uint32),
            known_value=np.array([0.0, 2.0]),
        )
        # Permute rows: old 0 -> new 2, old 1 -> new 0, old 2 -> new 1; reverse known order.
        perm = np.array([1, 2, 0])  # old_of_new
        inv = np.array([2, 0, 1], dtype=np.uint32)  # new_of_old
        A_perm = A[perm, :][:, perm].tocsc()
        f_perm = f[perm]
        row_perm = np.array([NONE_U32 if r == NONE_U32 else inv[r] for r in p1.row], dtype=np.uint32)
        p2 = AssembledProblemData(
            A=A_perm,
            f=f_perm,
            row=row_perm,
            known_node=np.array([5, 0], dtype=np.uint32),
            known_value=np.array([2.0, 0.0]),
        )
        comp = compare_assembled_problems(p1, p2)
        self.assertTrue(comp.equal, comp.reason)

        # Mutate one entry in A -> must report inequality
        A_bad = A_perm.tolil()
        A_bad[0, 0] += 1e-4
        p3 = AssembledProblemData(A=A_bad.tocsc(), f=f_perm, row=row_perm, known_node=p2.known_node, known_value=p2.known_value)
        self.assertFalse(compare_assembled_problems(p1, p3).equal)

        # Canonicalization idempotence
        c1 = canonicalize_assembled_problem(p1)
        c2 = canonicalize_assembled_problem(p2)
        np.testing.assert_array_equal(c1.row, c2.row)
        np.testing.assert_allclose(c1.f, c2.f)

    def test_locate_charges_graded_and_tie_breaking(self) -> None:
        # Tet 0: large coarse tet [0, 10]^3 corner; Tet 1: metal tet sharing face z=0; plus tiny dielectric tets
        # near (0.1, 0.1, 0.0) whose centroids are far closer to (0.1, 0.1, 0.05) than Tet 0's centroid!
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],    # 0
                [10.0, 0.0, 0.0],   # 1
                [0.0, 10.0, 0.0],   # 2
                [0.0, 0.0, 10.0],   # 3  -> Tet 0 centroid (2.5, 2.5, 2.5)
                [0.0, 0.0, -10.0],  # 4  -> Tet 1 (metal) shares face (0,1,2)
                [0.1, 0.1, -0.01],  # 5  -> tiny tet nearby
                [0.12, 0.1, -0.01], # 6
                [0.1, 0.12, -0.01], # 7
                [0.1, 0.1, -0.03],  # 8
            ],
            dtype=np.float64,
        )
        elements = np.array(
            [
                [0, 1, 2, 3],       # 0: dielectric
                [1, 0, 2, 4],       # 1: metal
                [5, 6, 7, 8],       # 2: tiny dielectric nearby
            ],
            dtype=np.uint32,
        )
        is_metal = np.array([False, True, False], dtype=bool)

        # Point deep in corner of huge tet 0, much closer to tet 2's centroid than tet 0's centroid
        e_ids, w = locate_charges(nodes, elements, is_metal, np.array([[0.1, 0.1, 0.05]]))
        self.assertEqual(int(e_ids[0]), 0)
        np.testing.assert_allclose(w[0] @ nodes[elements[0]], [0.1, 0.1, 0.05], atol=1e-13)

        # Point on shared face between metal tet 1 and dielectric tet 0 -> deterministically picks dielectric tet 0
        # Even if metal tet has LOWER index!
        elements_swapped = np.array([[1, 0, 2, 4], [0, 1, 2, 3]], dtype=np.uint32)
        is_metal_swapped = np.array([True, False], dtype=bool)
        e_face, _ = locate_charges(nodes, elements_swapped, is_metal_swapped, np.array([[1.0, 1.0, 0.0]]))
        self.assertEqual(int(e_face[0]), 1)

        # Point strictly inside metal tet -> error
        with self.assertRaisesRegex(ValueError, "metal"):
            locate_charges(nodes, elements, is_metal, np.array([[0.5, 0.5, -1.0]]))

        # Point outside mesh -> error
        with self.assertRaisesRegex(ValueError, "outside"):
            locate_charges(nodes, elements, is_metal, np.array([[50.0, 50.0, 50.0]]))

    def test_locate_metal_metal_face_next_to_huge_dielectric(self) -> None:
        # Two unit metal tets share the face x = 0; a 1e5 nm dielectric tet sits on M1's face
        # z = 0. A charge on the metal-metal face, 1e-10 nm inside the metal, is admitted by both
        # metal tets with min_i d_i = 0 (so neither holds it strictly) and by the dielectric
        # tet only through its ~1e-9 nm tolerance: it must still be rejected as inside metal.
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],        # 0 a
                [1.0, 0.0, 0.0],        # 1 b
                [0.0, 1.0, 0.0],        # 2 c
                [0.0, 0.0, 1.0],        # 3 d
                [-1.0, 0.0, 0.0],       # 4 e
                [0.25, 0.25, -1e5],     # 5 f (huge dielectric apex)
            ],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3], [0, 2, 3, 4], [0, 1, 2, 5]], dtype=np.uint32)
        is_metal = np.array([True, True, False])
        nb = compute_face_neighbours(elements)
        self.assertEqual(int(nb[0, 1]), 1)   # M1 face opposite b (x = 0) -> M2
        self.assertEqual(int(nb[0, 3]), 2)   # M1 face opposite d (z = 0) -> dielectric
        self.assertEqual(int(nb[1, 3]), 0)   # M2 face opposite e (x = 0) -> M1
        self.assertEqual(int(nb[2, 3]), 0)
        self.assertEqual(int(np.sum(nb != NONE_U32)), 4)
        p_in = np.array([[0.0, 0.3, 1e-10]])
        for given in (None, nb):
            with self.subTest(neighbours="computed" if given is None else "given"):
                with self.assertRaisesRegex(ValueError, "metal"):
                    locate_charges(nodes, elements, is_metal, p_in, neighbours=given)
                # Exactly on the metal-dielectric face z = 0 -> lands in the dielectric.
                e_ids, _ = locate_charges(nodes, elements, is_metal, np.array([[0.2, 0.3, 0.0]]), neighbours=given)
                self.assertEqual(int(e_ids[0]), 2)
        with self.assertRaisesRegex(ValueError, "neighbours must have shape"):
            locate_charges(nodes, elements, is_metal, p_in, neighbours=nb[:2])

    def test_locate_interface_vertex_with_metal_only_tets_lands_in_dielectric(self) -> None:
        # Metal cube [0,1]^3 next to a dielectric cube [1,2]x[0,1]^2. Vertex (1,1,1) lies on the
        # interface but also belongs to metal tets that touch the interface only at that vertex
        # (no dielectric face neighbour); the metal tets whose interface face passes through it
        # keep it on the interface, so it lands in the dielectric.
        mesh = _build_box_mesh(
            2, 1, 1, lx=2.0,
            material_fn=lambda cx, _cy, _cz: 0 if cx < 1.0 else 1,
            materials=[("metal", 1.0, 0.0, True, 0.0), ("diel", 1.0, 0.0, False, None)],
        )
        is_metal = mesh.material_is_metal[mesh.element_materials]
        for p in ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.5), (1.0, 0.25, 0.5)):
            with self.subTest(p=p):
                e_ids, _ = locate_charges(mesh.nodes, mesh.elements, is_metal, np.array([p]),
                                          neighbours=mesh.neighbours)
                self.assertFalse(bool(is_metal[int(e_ids[0])]))
        with self.assertRaisesRegex(ValueError, "metal"):
            locate_charges(mesh.nodes, mesh.elements, is_metal, np.array([[1.0 - 1e-9, 0.25, 0.5]]),
                           neighbours=mesh.neighbours)


    def test_comparator_row_scaled_and_symmetric(self) -> None:
        row = np.array([0, 1, NONE_U32], dtype=np.uint32)
        known_node = np.array([2], dtype=np.uint32)

        def prob(A, f, known=0.0, scale=1.0):
            return AssembledProblemData(
                A=sparse.csc_matrix(np.asarray(A, dtype=np.float64) * scale),
                f=np.asarray(f, dtype=np.float64) * scale,
                row=row,
                known_node=known_node,
                known_value=np.array([known], dtype=np.float64),
            )

        # A large row must not hide a wrong small row (explicit 0 vs 1 on the diagonal).
        big = prob([[1e12, 0.0], [0.0, 1.0]], [1.0, 1.0])
        bad_a = prob([[1e12, 0.0], [0.0, 0.0]], [1.0, 1.0])
        for x, y in ((big, bad_a), (bad_a, big)):
            comp = compare_assembled_problems(x, y)
            self.assertFalse(comp.equal)
            self.assertIn("canonical row 1", comp.reason)
            self.assertIn("A differs", comp.reason)

        # Same for f.
        f_ok = prob([[1e12, 0.0], [0.0, 1.0]], [1e12, 1.0])
        f_bad = prob([[1e12, 0.0], [0.0, 1.0]], [1e12, 0.0])
        for x, y in ((f_ok, f_bad), (f_bad, f_ok)):
            comp = compare_assembled_problems(x, y)
            self.assertFalse(comp.equal)
            self.assertIn("f differs", comp.reason)
            self.assertIn("canonical row 1", comp.reason)

        # Argument-order symmetry right at the tolerance boundary: |diff| = 1.5e-10 * max(a, b).
        a_ = prob([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        b_ = prob([[1.0 + 1.5e-10, 0.0], [0.0, 1.0]], [1.0, 1.0])
        self.assertEqual(compare_assembled_problems(a_, b_).equal, compare_assembled_problems(b_, a_).equal)
        self.assertFalse(compare_assembled_problems(a_, b_).equal)
        c_ = prob([[1.0 + 5e-11, 0.0], [0.0, 1.0]], [1.0, 1.0])
        self.assertTrue(compare_assembled_problems(a_, c_).equal)
        self.assertTrue(compare_assembled_problems(c_, a_).equal)

        # Tiny-scale systems are judged like unit-scale ones (atol defaults to 0).
        tiny = prob([[2.0, -1.0], [-1.0, 2.0]], [1.0, 3.0], scale=1e-14)
        tiny_close = prob([[2.0 * (1 + 1e-12), -1.0], [-1.0, 2.0]], [1.0, 3.0 * (1 + 1e-12)], scale=1e-14)
        tiny_far = prob([[2.0 * (1 + 1e-6), -1.0], [-1.0, 2.0]], [1.0, 3.0], scale=1e-14)
        self.assertTrue(compare_assembled_problems(tiny, tiny_close).equal)
        self.assertFalse(compare_assembled_problems(tiny, tiny_far).equal)
        tiny_f_far = prob([[2.0, -1.0], [-1.0, 2.0]], [1.0, 3.0 * (1 + 1e-6)], scale=1e-14)
        self.assertFalse(compare_assembled_problems(tiny, tiny_f_far).equal)

        # Dirichlet-lifting scale: f entries that cancel to ~0 are judged against s_i * U.
        lift1 = prob([[1.0, 0.0], [0.0, 1.0]], [1e-3, 1.0], known=1e6)
        lift2 = prob([[1.0, 0.0], [0.0, 1.0]], [1e-3 + 1e-5, 1.0], known=1e6)
        self.assertTrue(compare_assembled_problems(lift1, lift2).equal)

        # Non-finite values never compare equal (NaN fails every <= test silently otherwise).
        nan_f = prob([[1.0, 0.0], [0.0, 1.0]], [1.0, np.nan])
        for x, y in ((nan_f, nan_f), (a_, nan_f), (nan_f, a_)):
            comp = compare_assembled_problems(x, y)
            self.assertFalse(comp.equal)
            self.assertIn("non-finite values in f", comp.reason)
        inf_a = prob([[np.inf, 0.0], [0.0, 1.0]], [1.0, 1.0])
        self.assertFalse(compare_assembled_problems(inf_a, inf_a).equal)

        # s_i * U = 1e310 overflows: the f check must not turn into |diff| <= inf.
        one_row = np.array([0, NONE_U32], dtype=np.uint32)

        def hi(f: float) -> AssembledProblemData:
            return AssembledProblemData(
                A=sparse.csc_matrix(np.array([[1e160]])), f=np.array([f]), row=one_row,
                known_node=np.array([1], dtype=np.uint32), known_value=np.array([1e150]),
            )

        for x, y in ((hi(1.0), hi(1e306)), (hi(1e306), hi(1.0))):
            comp = compare_assembled_problems(x, y)
            self.assertFalse(comp.equal)
            self.assertIn("f differs", comp.reason)
        self.assertTrue(compare_assembled_problems(hi(1.0), hi(1.0 + 1e140)).equal)   # within rtol * s_i * U
        self.assertFalse(compare_assembled_problems(hi(-1.7e308), hi(1.7e308)).equal)  # |diff| overflows

        # Known values per node, symmetric.
        k1 = prob([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], known=1.0)
        k2 = prob([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], known=1.0 + 1e-6)
        comp = compare_assembled_problems(k1, k2)
        self.assertFalse(comp.equal)
        self.assertIn("known values differ (worst at node 2", comp.reason)

    def test_locate_signed_distance_large_coordinates(self) -> None:
        # Huge (1e6 nm) dielectric tet next to a tiny (1e-6 nm) metal tet: a charge halfway
        # inside the metal must be rejected, not swallowed by the dielectric's tolerance.
        h = 1e-6
        x0 = y0 = 1e5
        nodes = np.array(
            [
                [0.0, 0.0, 0.0], [1e6, 0.0, 0.0], [0.0, 1e6, 0.0], [0.0, 0.0, 1e6],    # dielectric
                [x0, y0, 0.0], [x0 + h, y0, 0.0], [x0, y0 + h, 0.0], [x0, y0, -h],   # metal
            ],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint32)
        is_metal = np.array([False, True])
        centroid = nodes[4:8].mean(axis=0)
        with self.assertRaisesRegex(ValueError, "metal"):
            locate_charges(nodes, elements, is_metal, centroid[None, :])
        # allow_metal=True locates it in the metal tet.
        e_ids, w = locate_charges(nodes, elements, is_metal, centroid[None, :], allow_metal=True)
        self.assertEqual(int(e_ids[0]), 1)
        # Node coordinates 1e5 + 1e-6 carry ~1e-5 relative rounding, so weights are only that exact.
        np.testing.assert_allclose(w[0], 0.25, atol=1e-4)

        # A charge on a slanted metal-dielectric interface at x ~ 1e6 (not exactly
        # representable on the plane) must land in the dielectric, even with the metal tet first.
        a = np.array([1e6, 0.0, 0.0])
        b = np.array([1e6 - 1.0, 1.0, 0.0])
        c = np.array([1e6 - 1.0, 0.0, 1.0])
        n = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        face_c = (a + b + c) / 3.0
        nodes2 = np.array([a, b, c, face_c + 0.5 * n, face_c - 0.5 * n], dtype=np.float64)
        elements2 = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.uint32)   # metal first
        is_metal2 = np.array([True, False])
        p_on = (0.2 * a + 0.3 * b + 0.5 * c)[None, :]
        e_ids, w = locate_charges(nodes2, elements2, is_metal2, p_on)
        self.assertEqual(int(e_ids[0]), 1)
        np.testing.assert_allclose(w[0, :3], [0.2, 0.3, 0.5], atol=1e-6)
        self.assertLess(w[0, 3], 1e-6)
        # Clearly inside the metal side (1e-3 nm deep) -> rejected.
        with self.assertRaisesRegex(ValueError, "metal"):
            locate_charges(nodes2, elements2, is_metal2, p_on + 1e-3 * n)
        # Clearly outside both -> outside.
        with self.assertRaisesRegex(ValueError, "outside"):
            locate_charges(nodes2, elements2, is_metal2, (face_c + 2.0 * n)[None, :])

    def test_locate_huge_right_tet_does_not_overflow(self) -> None:
        # |cross| of unnormalized edges ~ L^2 would overflow its squares at L = 1e80; the
        # per-tet shift + scale keeps every intermediate O(1).
        for length in (1e25, 1e80, 1e-25):
            with self.subTest(length=length):
                nodes = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]])
                elements = np.array([[0, 1, 2, 3]], dtype=np.uint32)
                p = np.array([[0.1, 0.2, 0.3]]) * length
                with np.errstate(all="raise"):
                    e_ids, w = locate_charges(nodes, elements, np.array([False]), p)
                self.assertEqual(int(e_ids[0]), 0)
                np.testing.assert_allclose(w[0], [0.4, 0.1, 0.2, 0.3], rtol=1e-12)
        # Sol's case: locate (1, 1, 1) in a right tet with edge length 1e80 (direct API).
        nodes = np.array([[0.0, 0.0, 0.0], [1e80, 0.0, 0.0], [0.0, 1e80, 0.0], [0.0, 0.0, 1e80]])
        with np.errstate(all="raise"):
            e_ids, w = locate_charges(nodes, np.array([[0, 1, 2, 3]], dtype=np.uint32), np.array([False]),
                                      np.array([[1.0, 1.0, 1.0]]))
        self.assertEqual(int(e_ids[0]), 0)
        self.assertTrue(np.all(np.isfinite(w)))
        np.testing.assert_allclose(w[0], [1.0, 1e-80, 1e-80, 1e-80], rtol=1e-12)
        # Outside it is still outside.
        with self.assertRaisesRegex(ValueError, "outside"):
            locate_charges(nodes, np.array([[0, 1, 2, 3]], dtype=np.uint32), np.array([False]),
                           np.array([[-1e70, 1.0, 1.0]]))

    def test_strict_h5_validation_rejects_malformed_files(self) -> None:
        mesh = _build_box_mesh(2, 1, 1, lx=2.0, screened=False)
        good_mesh = self.tmp / "good_mesh.h5"
        write_mesh_properties(good_mesh, mesh)
        read_mesh_properties(good_mesh)

        ap = AssembledProblemData(
            A=sparse.csc_matrix(np.array([[2.0, -1.0, 0.0], [-1.0, 2.5, 0.5], [0.0, 0.5, 4.0]])),
            f=np.array([1.0, -2.5, 0.0]),
            row=np.array([NONE_U32, 0, 1, 2], dtype=np.uint32),
            known_node=np.array([0], dtype=np.uint32),
            known_value=np.array([1.0]),
        )
        good_ap = self.tmp / "good_ap.h5"
        write_assembled_problem(good_ap, ap)
        read_assembled_problem(good_ap)

        def set_attr_u32(name, value):
            return lambda h5: h5.attrs.create(name, np.uint32(value), dtype="<u4")

        # (name, source file, mutation, Python error regex, also rejected by the C++ reader?)
        mesh_cases = [
            ("bc_id_99", lambda h5: _replace_dataset(
                h5, "boundary_elements/boundary_condition",
                np.full(h5["boundary_elements/boundary_condition"].shape, 99, dtype="<u2")),
             "boundary_condition references out-of-range", True),
            ("bad_bool", lambda h5: _replace_dataset(
                h5, "materials/is_metal", np.array([2], dtype="<u1")), "invalid boolean flag", True),
            ("bad_has_alpha", lambda h5: _replace_dataset(
                h5, "boundary_conditions/has_alpha", np.array([0, 7], dtype="<u1")), "invalid boolean flag", True),
            ("wrong_rank", lambda h5: _replace_dataset(
                h5, "nodes", np.asarray(h5["nodes"][()]).reshape(-1)), "rank", True),
            ("wrong_type", lambda h5: _replace_dataset(
                h5, "elements/node", np.asarray(h5["elements/node"][()]).astype("<i8")), "type", True),
            ("bad_bc_type", lambda h5: _replace_dataset(
                h5, "boundary_conditions/type", np.array([0, 3], dtype="<u1")), "BcType", True),
            ("bad_node_idx", lambda h5: _replace_dataset(
                h5, "elements/node",
                np.where(np.arange(h5["elements/node"].size).reshape(-1, 4) == 0, 1000,
                         np.asarray(h5["elements/node"][()])).astype("<u4")), "out-of-range node", True),
            ("bad_neighbour", lambda h5: _replace_dataset(
                h5, "neighbours", np.full(h5["neighbours"].shape, 500, dtype="<u4")), "out-of-range element", True),
            ("bad_screened", set_attr_u32("screened", 2), "screened", True),
            ("screened_u8", lambda h5: h5.attrs.create("screened", np.uint8(1), dtype="<u1"), "uint32", True),
            ("screened_31bit", _set_31bit_screened, "uint32", True),
            ("wrapped_name_offsets", _wrap_name_offsets, "invalid materials/name_offsets", True),
            ("nan_node", lambda h5: _replace_dataset(
                h5, "nodes", np.where(np.arange(h5["nodes"].size).reshape(-1, 3) == 0, np.nan,
                                      np.asarray(h5["nodes"][()]))), "non-finite", False),
        ]
        ap_cases = [
            ("unsorted_inner", lambda h5: _replace_dataset(
                h5, "A/inner_index", np.asarray(h5["A/inner_index"][()])[[1, 0, 2, 3, 4, 5, 6]].astype("<i4")),
             "strictly increasing", True),
            ("inner_oob", lambda h5: _replace_dataset(
                h5, "A/inner_index", np.array([0, 1, 0, 1, 2, 1, 9], dtype="<i4")), "out-of-range row", True),
            ("outer_bad_end", lambda h5: _replace_dataset(
                h5, "A/outer_index", np.array([0, 2, 5, 6], dtype="<i4")), "start at 0 and end at nnz", True),
            ("rows_u32", lambda h5: h5["A"].attrs.create("rows", np.uint32(3), dtype="<u4"), "uint64", True),
            ("row_oob", lambda h5: _replace_dataset(
                h5, "row", np.array([NONE_U32, 0, 1, 7], dtype="<u4")), "out-of-range row", True),
            ("known_oob", lambda h5: _replace_dataset(
                h5, "known/node", np.array([40], dtype="<u4")), "known/node references out-of-range", True),
            ("f_wrong_rank", lambda h5: _replace_dataset(
                h5, "f", np.asarray(h5["f"][()]).reshape(1, 3)), "rank", True),
            ("inf_value", lambda h5: _replace_dataset(
                h5, "A/values", np.where(np.arange(7) == 0, np.inf, np.asarray(h5["A/values"][()]))),
             "non-finite", False),
        ]

        try:
            tool = resolve_binary("io_roundtrip_tool")
        except FileNotFoundError:
            tool = None

        for kind, good, cases, reader in (
            ("mesh", good_mesh, mesh_cases, read_mesh_properties),
            ("assembled", good_ap, ap_cases, read_assembled_problem),
        ):
            if tool is not None:
                ok = subprocess.run([str(tool), kind, str(good), str(self.tmp / f"rt_{kind}.h5")],
                                    capture_output=True, text=True)
                self.assertEqual(ok.returncode, 0, ok.stderr)
            for name, mutate, regex, cpp_rejects in cases:
                with self.subTest(kind=kind, case=name):
                    bad = _mutate_h5(good, self.tmp / f"bad_{kind}_{name}.h5", mutate)
                    with self.assertRaisesRegex(RuntimeError, regex):
                        reader(bad)
                    if tool is not None and cpp_rejects:
                        res = subprocess.run([str(tool), kind, str(bad), str(self.tmp / "rt_bad.h5")],
                                             capture_output=True, text=True)
                        self.assertNotEqual(res.returncode, 0, f"C++ reader accepted {name}")

        # Nonzero magnitudes outside the tier's range are rejected by the readers: mesh inputs
        # [1e-30, 1e30], coordinates |v| <= 1e30, AssembledProblem values [1e-100, 1e100].
        mesh_range, problem_range = r"\[1e-30, 1e30\]", r"\[1e-100, 1e100\]"
        coord_range = r"\|v\| <= 1e30"
        for kind, good, name, reader, value, regex in (
            ("mesh", good_mesh, "nodes", read_mesh_properties, 1e31, coord_range),
            ("mesh", good_mesh, "nodes", read_mesh_properties, -1e200, coord_range),
            ("mesh", good_mesh, "materials/eps_r", read_mesh_properties, 1e-40, mesh_range),
            ("mesh", good_mesh, "materials/eps_r", read_mesh_properties, 1e40, mesh_range),
            ("mesh", good_mesh, "boundary_conditions/value", read_mesh_properties, 1e31, mesh_range),
            ("assembled", good_ap, "f", read_assembled_problem, 1e-200, problem_range),
            ("assembled", good_ap, "A/values", read_assembled_problem, 1e101, problem_range),
            ("assembled", good_ap, "known/value", read_assembled_problem, 1e-200, problem_range),
        ):
            with self.subTest(kind=kind, out_of_range=name, value=value):
                def put(h5, name=name, value=value):
                    vals = np.asarray(h5[name][()]).copy()
                    vals.reshape(-1)[0] = value
                    _replace_dataset(h5, name, vals)
                bad = _mutate_h5(good, self.tmp / f"range_{kind}_{name.replace('/', '_')}_{value:g}.h5", put)
                with self.assertRaisesRegex(ValueError, rf"{name}.*{regex}"):
                    reader(bad)
        # Rounding-noise coordinates next to zero (refined meshes produce e.g. -7.5e-32) are fine.
        for tiny in (-7.5e-32, 1e-200, 5e-324):
            with self.subTest(tiny_coordinate=tiny):
                def put_tiny(h5, tiny=tiny):
                    vals = np.asarray(h5["nodes"][()]).copy()
                    vals[0, 0] = tiny
                    _replace_dataset(h5, "nodes", vals)
                ok_mesh = _mutate_h5(good_mesh, self.tmp / f"tiny_coord_{tiny:g}.h5", put_tiny)
                self.assertEqual(float(read_mesh_properties(ok_mesh).nodes[0, 0]), tiny)
        # Values outside the mesh tier but inside the problem tier are fine in an AssembledProblem.
        ok_ap = _mutate_h5(good_ap, self.tmp / "range_ap_ok.h5", lambda h5: _replace_dataset(
            h5, "f", np.full_like(np.asarray(h5["f"][()]), 1e90)))
        np.testing.assert_array_equal(read_assembled_problem(ok_ap).f, 1e90)

        # SolverResult / NodalField readers.
        sr = self.tmp / "sr.h5"
        write_solver_result(sr, SolverResultData(x=np.array([1.0, 2.0])))
        with self.assertRaisesRegex(RuntimeError, "rank"):
            read_solver_result(_mutate_h5(sr, self.tmp / "sr_rank.h5",
                                          lambda h5: _replace_dataset(h5, "x", np.ones((2, 1)))))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            read_solver_result(_mutate_h5(sr, self.tmp / "sr_nan.h5",
                                          lambda h5: _replace_dataset(h5, "x", np.array([1.0, np.nan]))))
        # SolverResult::x only needs to be finite (the field stage copies it without arithmetic).
        for x_ok in ([1.0, 1e200], [1e-300, -1e300], [5e-324, 0.0]):
            with self.subTest(finite_x=x_ok):
                sr_ok = self.tmp / f"sr_ok_{x_ok[0]:g}.h5"
                write_solver_result(sr_ok, SolverResultData(x=np.array(x_ok)))
                np.testing.assert_array_equal(read_solver_result(sr_ok).x, x_ok)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            read_solver_result(_mutate_h5(sr, self.tmp / "sr_inf.h5",
                                          lambda h5: _replace_dataset(h5, "x", np.array([1.0, -np.inf]))))
        nf = self.tmp / "nf.h5"
        write_nodal_field(nf, NodalFieldData(
            u=np.zeros(3), point_source_position=np.zeros((1, 3)), point_source_value=np.ones(1),
            point_source_element=np.zeros(1, dtype=np.uint32)))
        read_nodal_field(nf)
        with self.assertRaisesRegex(RuntimeError, "columns"):
            read_nodal_field(_mutate_h5(nf, self.tmp / "nf_cols.h5", lambda h5: _replace_dataset(
                h5, "point_sources/position", np.zeros((1, 2)))))
        with self.assertRaisesRegex(RuntimeError, "type"):
            read_nodal_field(_mutate_h5(nf, self.tmp / "nf_type.h5", lambda h5: _replace_dataset(
                h5, "point_sources/element", np.zeros(1, dtype="<i4"))))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            read_nodal_field(_mutate_h5(nf, self.tmp / "nf_inf.h5", lambda h5: _replace_dataset(
                h5, "point_sources/value", np.array([np.inf]))))

        # In-memory MeshData with an out-of-range BC id is rejected by assembly too.
        if HAS_SKFEM:
            bad_mem = _replace_mesh(mesh, boundary_element_bcs=np.full_like(mesh.boundary_element_bcs, 99))
            with self.assertRaisesRegex(ValueError, "boundary condition 99"):
                assemble_problem(bad_mem)

    def test_solver_rhs_scaling_and_convergence_warning(self) -> None:
        def one(f: float) -> AssembledProblemData:
            return AssembledProblemData(
                A=sparse.csc_matrix(np.array([[1.0]])), f=np.array([f]),
                row=np.array([0], dtype=np.uint32), known_node=np.zeros(0, dtype=np.uint32),
                known_value=np.zeros(0),
            )

        methods = ["direct", "amg"] if HAS_PYAMG else ["direct"]
        for method in methods:
            for f in (1e-90, 1e90, -3.0):
                with self.subTest(method=method, f=f):
                    res, info = solve_problem(one(f), method=method)
                    self.assertEqual(float(res.x[0]), f)
                    self.assertLess(info.residual, 1e-12)
                    self.assertEqual(info.residual, 0.0)
                    self.assertEqual(info.warnings, [])
        # Nonzero magnitudes outside [1e-100, 1e100] (incl. subnormals) are rejected up front.
        for f in (1e-310, 5e-324, 1e170, -1e-101):
            with self.subTest(out_of_range=f), self.assertRaisesRegex(ValueError, r"\[1e-100, 1e100\]"):
                solve_problem(one(f), method="direct")
        bad_a = AssembledProblemData(
            A=sparse.csc_matrix(np.array([[1e-310]])), f=np.array([1.0]),
            row=np.array([0], dtype=np.uint32), known_node=np.zeros(0, dtype=np.uint32), known_value=np.zeros(0),
        )
        with self.assertRaisesRegex(ValueError, r"A\.data .*\[1e-100, 1e100\]"):
            solve_problem(bad_a, method="direct")
        if HAS_PYAMG:
            # CG converges in its only iteration: SciPy reports info=maxiter, but no warning.
            res, info = solve_problem(one(1.0), method="amg", maxiter=1)
            self.assertEqual(float(res.x[0]), 1.0)
            self.assertEqual(info.warnings, [])

            # Genuine non-convergence still warns, judged by the true residual.
            n = 200
            A = sparse.diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csc")
            hard = AssembledProblemData(A=A, f=np.ones(n), row=np.arange(n, dtype=np.uint32),
                                        known_node=np.zeros(0, dtype=np.uint32), known_value=np.zeros(0))
            _, info = solve_problem(hard, method="amg", rtol=1e-30, maxiter=1)
            self.assertEqual(len(info.warnings), 1)
            self.assertGreater(info.residual, 1e-30)

            # The caller's global RNG state is untouched by the seeded PyAMG setup.
            np.random.seed(1234)
            expected = np.random.rand()
            np.random.seed(1234)
            solve_problem(hard, method="amg", seed=7)
            self.assertEqual(np.random.rand(), expected)

        zero, zinfo = solve_problem(one(0.0), method="direct")
        self.assertEqual(float(zero.x[0]), 0.0)
        self.assertEqual(zinfo.residual, 0.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            solve_problem(one(float("nan")))
        with self.assertRaisesRegex(ValueError, "maxiter must be an integer"):
            solve_problem(one(1.0), maxiter=2.5)
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            solve_problem(one(1.0), seed=True)

    @unittest.skipUnless(HAS_SKFEM, "scikit-fem not installed")
    def test_strict_cli_and_api_scalar_validation(self) -> None:
        mesh_h5 = _box_mesh_h5(self.tmp, nx=2, ny=1, nz=1)
        out = self.tmp / "out.h5"
        for value, why in ((2.9, "intorder must be an integer"), (True, "intorder must be an integer")):
            cfg = self.tmp / "cfg.json"
            cfg.write_text(json.dumps({"intorder": value}), encoding="utf-8")
            code, err = _run_cli(["assemble", "--mesh", str(mesh_h5), "--config", str(cfg), "--out", str(out)])
            self.assertEqual(code, 1)
            self.assertIn(why, err)
        code, err = _run_cli(["assemble", "--mesh", str(mesh_h5), "--constant-source", "nan", "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertIn("constant_source must be finite", err)
        self.assertFalse(out.exists())

        code, _ = _run_cli(["assemble", "--mesh", str(mesh_h5), "--out", str(out)])
        self.assertEqual(code, 0)
        cfg = self.tmp / "solve.json"
        for bad in ({"maxiter": 10.0}, {"seed": "1"}, {"direct_threshold": False}, {"rtol": "1e-3"}):
            cfg.write_text(json.dumps(bad), encoding="utf-8")
            code, err = _run_cli(["solve", "--in", str(out), "--config", str(cfg), "--out", str(self.tmp / "s.h5")])
            self.assertEqual(code, 1, bad)
        cfg.write_text(json.dumps({"seed": 3, "method": "direct"}), encoding="utf-8")
        code, err = _run_cli(["solve", "--in", str(out), "--config", str(cfg), "--out", str(self.tmp / "s.h5")])
        self.assertEqual(code, 0, err)

        mesh = read_mesh_properties(mesh_h5)
        with self.assertRaisesRegex(ValueError, "intorder must be an integer"):
            assemble_problem(mesh, intorder=2.9)
        with self.assertRaisesRegex(ValueError, "intorder must be an integer"):
            assemble_problem(mesh, intorder=True)
        with self.assertRaisesRegex(ValueError, "constant_source must be finite"):
            assemble_problem(mesh, constant_source=float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            assemble_problem(mesh, source=lambda x, _e: np.full(x.shape[1:], np.nan))
        for bad in (complex(1.0, math.inf), 1.0 + 1.0j):
            with self.subTest(source=bad), self.assertRaisesRegex(ValueError, "real floating-point"):
                assemble_problem(mesh, source=lambda x, _e, bad=bad: np.full(x.shape[1:], bad))
        with self.assertRaisesRegex(ValueError, "real floating-point"):
            assemble_problem(mesh, source=lambda x, _e: np.full(x.shape[1:], 1.0, dtype=object))
        with self.assertRaisesRegex(ValueError, r"source\(\.\.\.\) output .*\[1e-30, 1e30\]"):
            assemble_problem(mesh, source=lambda x, _e: np.full(x.shape[1:], 1e-300))
        with self.assertRaisesRegex(ValueError, r"source\(\.\.\.\) output .*\[1e-30, 1e30\]"):
            assemble_problem(mesh, source=lambda x, _e: np.full(x.shape[1:], 1e31))
        with self.assertRaisesRegex(ValueError, r"material_eps_r .*\[1e-30, 1e30\]"):
            assemble_problem(_replace_mesh(mesh, material_eps_r=np.array([1e-310])))
        with self.assertRaisesRegex(ValueError, r"material_eps_r .*\[1e-30, 1e30\]"):
            assemble_problem(_replace_mesh(mesh, material_eps_r=np.array([1e95])))
        with self.assertRaisesRegex(ValueError, r"constant_source .*\[1e-30, 1e30\]"):
            assemble_problem(mesh, constant_source=1e150)
        with self.assertRaisesRegex(ValueError, r"constant_source .*\[1e-30, 1e30\]"):
            assemble_problem(mesh, constant_source=1e-31)
        with self.assertRaisesRegex(ValueError, "must return shape"):
            assemble_problem(mesh, source=lambda x, _e: np.ones(3))

@unittest.skipUnless(HAS_CORE_DEPS, CORE_DEPS_SKIP_REASON)
class TestStageEnvironmentProbe(unittest.TestCase):
    """The dependency probe follows the stage's environment; no stale per-process memo."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # A PYTHONPATH entry whose metadata shadows the installed numpy version.
        dist = self.tmp / "fake_site" / "numpy-0.0.1.dist-info"
        dist.mkdir(parents=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: numpy\nVersion: 0.0.1\n", encoding="utf-8")
        self.fake_site = str(dist.parent)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pythonpath_and_stage_env_refresh_the_probe_and_stage_key(self) -> None:
        env_without = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        with mock.patch.dict(os.environ, env_without, clear=True):
            plain = make_solve_stage().version
            self.assertNotIn("numpy=0.0.1", plain)
            # Per-stage env override reaches the probe.
            self.assertIn("numpy=0.0.1", make_solve_stage(env={"PYTHONPATH": self.fake_site}).version)
            # So does an inherited PYTHONPATH change in the same process (no memoized probe).
            with mock.patch.dict(os.environ, {"PYTHONPATH": self.fake_site}):
                self.assertIn("numpy=0.0.1", make_solve_stage().version)
            self.assertEqual(make_solve_stage().version, plain)

            class _Run:
                """Minimal RunContext stand-in: real `get_stage` memo, `run_stage` returns the stage."""

                def __init__(self) -> None:
                    self.stages: dict = {}

                def get_stage(self, key, factory):
                    if key not in self.stages:
                        self.stages[key] = factory(None)
                    return self.stages[key]

                def run_stage(self, stage, **_kwargs):
                    return stage

            run = _Run()
            s_plain = stage_solve(run, "p.h5")
            s_env = stage_solve(run, "p.h5", env={"PYTHONPATH": self.fake_site})
            self.assertIsNot(s_plain, s_env)
            self.assertNotIn("numpy=0.0.1", s_plain.version)
            self.assertIn("numpy=0.0.1", s_env.version)
            self.assertIs(stage_solve(run, "p.h5", env={"PYTHONPATH": self.fake_site}), s_env)
            self.assertIs(stage_solve(run, "p.h5"), s_plain)
            self.assertEqual(len(run.stages), 2)
            cache = Cache(self.tmp / "cache")
            self.assertNotEqual(s_plain.compute_code_hash(cache), s_env.compute_code_hash(cache))

            # Mutating the inherited PYTHONPATH inside the same RunContext must not reuse the
            # memoized Stage: a fresh probe and a distinct request key.
            with mock.patch.dict(os.environ, {"PYTHONPATH": self.fake_site}):
                s_inherited = stage_solve(run, "p.h5")
                self.assertIsNot(s_inherited, s_plain)
                self.assertIn("numpy=0.0.1", s_inherited.version)
                self.assertIs(stage_solve(run, "p.h5"), s_inherited)
            self.assertIs(stage_solve(run, "p.h5"), s_plain)
            self.assertEqual(len(run.stages), 3)

            def _request_key(stage) -> str:
                return Cache.compute_request_key(stage.name, stage.compute_code_hash(cache), {"in": "p"}, {})

            self.assertNotEqual(_request_key(s_inherited), _request_key(s_plain))


@unittest.skipUnless(HAS_CORE_DEPS, CORE_DEPS_SKIP_REASON)
@unittest.skipUnless(HAS_SKFEM, "scikit-fem not installed")
class TestAssemblyAndSolvers(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_closed_form_element_matrix_crosscheck(self) -> None:
        """20-line closed-form NumPy check against scikit-fem stiffness, mass, and Robin face mass."""
        from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, MeshTet
        from skfem.helpers import dot, grad

        rng = np.random.default_rng(42)
        # Single random non-degenerate tet
        pts = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, -0.2], [0.2, 1.3, 0.15], [-0.1, 0.25, 0.95]])
        pts += rng.uniform(-0.05, 0.05, size=pts.shape)
        t = np.array([[0, 1, 2, 3]], dtype=np.int64)
        m = MeshTet(pts.T, t.T)
        self.assertTrue(np.array_equal(m.p, pts.T))

        eps_val, c_val, alpha_val = 3.7, 2.4, 0.85
        b = Basis(m, ElementTetP1(), intorder=2)

        @BilinearForm
        def form(u, v, w):
            return w.eps * dot(grad(u), grad(v)) + w.c * u * v

        K_sk = form.assemble(b, eps=np.array([[eps_val]]), c=np.array([[c_val]])).toarray()

        # Closed-form P1 tet stiffness + mass:
        # [1 x_i y_i z_i] @ coeffs = I_4 => gradients are rows 1..3 of coeffs^T
        M_coord = np.hstack([np.ones((4, 1)), pts])
        vol = abs(np.linalg.det(M_coord)) / 6.0
        G = np.linalg.inv(M_coord)[1:, :].T  # (4, 3) gradients of phi_0..phi_3
        K_exact = eps_val * vol * (G @ G.T) + c_val * (vol / 20.0) * (np.ones((4, 4)) + np.eye(4))
        np.testing.assert_allclose(K_sk, K_exact, rtol=1e-13, atol=1e-13)

        # Closed-form Robin face mass on face (0, 1, 2)
        face_idx = None
        for fi in range(m.facets.shape[1]):
            if tuple(sorted(m.facets[:, fi].tolist())) == (0, 1, 2):
                face_idx = fi
                break
        assert face_idx is not None
        fb = FacetBasis(m, ElementTetP1(), facets=np.array([face_idx]), intorder=2)

        @BilinearForm
        def rform(u, v, w):
            return w.alpha * u * v

        R_sk = rform.assemble(fb, alpha=np.array([[alpha_val]])).toarray()
        area = 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
        R_exact = np.zeros((4, 4))
        R_exact[:3, :3] = alpha_val * (area / 12.0) * (np.ones((3, 3)) + np.eye(3))
        np.testing.assert_allclose(R_sk, R_exact, rtol=1e-13, atol=1e-13)

    def test_linear_exact_dirichlet_neumann_robin(self) -> None:
        # u(x) = 2 + 3 x satisfies -div(eps grad u) = 0, eps = 4.0:
        # at x=0 (n=-1): eps * du/dn = -12; with Robin alpha=2: eps*du/dn + 2*u = -12 + 4 = -8
        # at x=2 (n=+1): eps * du/dn = +12 (Neumann g = 12)
        mesh = _build_box_mesh(
            4, 2, 2,
            lx=2.0,
            materials=[("diel", 4.0, 0.0, False, None)],
            left_bc=(BC_ROBIN, -8.0, 2.0),
            right_bc=(BC_NEUMANN, 12.0, None),
            screened=False,
        )
        prob, info = assemble_problem(mesh)
        self.assertEqual(info.n_dirichlet, 0)
        res, sinfo = solve_problem(prob, method="direct")
        self.assertLess(sinfo.residual, 1e-13)
        field, _ = build_nodal_field(mesh, prob, res)
        u_exact = 2.0 + 3.0 * mesh.nodes[:, 0]
        np.testing.assert_allclose(field.u, u_exact, rtol=1e-12, atol=1e-12)

    def test_constant_and_callable_smooth_sources(self) -> None:
        # 1. Constant source: -eps * u''(x) = f0, u(0) = u(L) = 0 => u(x) = (f0 / (2*eps)) * x * (L - x).
        # On a uniform structured mesh, 1D quadratic nodal values are superconvergent (exact to ~1e-14).
        L, eps, f0 = 2.0, 3.0, 6.0
        mesh = _build_box_mesh(
            6, 2, 2,
            lx=L,
            materials=[("diel", eps, 0.0, False, None)],
            left_bc=(BC_DIRICHLET, 0.0, None),
            right_bc=(BC_DIRICHLET, 0.0, None),
            screened=False,
        )
        prob, _ = assemble_problem(mesh, constant_source=f0)
        res, _ = solve_problem(prob, method="direct")
        field, _ = build_nodal_field(mesh, prob, res)
        x = mesh.nodes[:, 0]
        np.testing.assert_allclose(field.u, (f0 / (2.0 * eps)) * x * (L - x), rtol=1e-12, atol=1e-12)

        # 2. Smooth callable source: u(x) = sin(pi x / L) => f(x) = eps * (pi/L)^2 * sin(pi x / L).
        # Check O(h^2) convergence under uniform 3D mesh refinement (h -> h/2).
        errors = []
        for k in (2, 4):
            m_n = _build_box_mesh(
                4 * k, 2 * k, 2 * k,
                lx=L,
                materials=[("diel", eps, 0.0, False, None)],
                left_bc=(BC_DIRICHLET, 0.0, None),
                right_bc=(BC_DIRICHLET, 0.0, None),
                screened=False,
            )
            p_n, _ = assemble_problem(
                m_n,
                source=lambda coords, _e: eps * (math.pi / L) ** 2 * np.sin(math.pi * coords[0] / L),
                intorder=4,
            )
            r_n, _ = solve_problem(p_n, method="direct")
            f_n, _ = build_nodal_field(m_n, p_n, r_n)
            u_ex = np.sin(math.pi * m_n.nodes[:, 0] / L)
            errors.append(float(np.max(np.abs(f_n.u - u_ex))))
        self.assertGreater(errors[0] / errors[1], 3.3)
        self.assertLess(errors[1], 0.005)

    def test_screening_on_vs_off(self) -> None:
        # Pure Neumann g=0 on right, g=1 on left: only well-posed when screened=True!
        mesh_on = _build_box_mesh(
            4, 2, 2,
            lx=2.0,
            materials=[("diel", 2.0, 1.0, False, None)],
            left_bc=(BC_NEUMANN, 1.0, None),
            right_bc=(BC_NEUMANN, 0.0, None),
            screened=True,
        )
        prob, _ = assemble_problem(mesh_on)
        res, _ = solve_problem(prob, method="direct")
        field, _ = build_nodal_field(mesh_on, prob, res)
        # Solution u(x) = cosh(L - x) / (eps * sinh(L)) (since k = 1/lambda = 1, -eps u'(0) = 1)
        L = 2.0
        u_exact = np.cosh(L - mesh_on.nodes[:, 0]) / (2.0 * np.sinh(L))
        np.testing.assert_allclose(field.u, u_exact, atol=0.02)

        # Turning screening off leaves pure Neumann -> singular system error
        mesh_off = _build_box_mesh(
            4, 2, 2,
            lx=2.0,
            materials=[("diel", 2.0, 1.0, False, None)],
            left_bc=(BC_NEUMANN, 1.0, None),
            right_bc=(BC_NEUMANN, 0.0, None),
            screened=False,
        )
        with self.assertRaisesRegex(ValueError, "singular"):
            assemble_problem(mesh_off)

    def test_floating_and_prescribed_metals(self) -> None:
        # 3-layer stack along x in [0, 3]:
        # [0, 1]: diel eps=2, [1, 2]: floating metal, [2, 3]: diel eps=6
        # u(0) = 0V, u(3) = 4V => zero net flux across metal: eps1*(V_m - 0)/1 = eps2*(4 - V_m)/1 => 2 V_m = 6(4 - V_m) => V_m = 3.0 V!
        materials = [
            ("diel1", 2.0, 0.0, False, None),
            ("metal", 1.0, 1.0, True, None),  # screening_length=1.0 on metal MUST be ignored!
            ("diel2", 6.0, 0.0, False, None),
        ]

        def mat_fn(cx: float, _cy: float, _cz: float) -> int:
            if cx < 1.0:
                return 0
            if cx < 2.0:
                return 1
            return 2

        mesh = _build_box_mesh(
            6, 2, 2,
            lx=3.0,
            material_fn=mat_fn,
            materials=materials,
            left_bc=(BC_DIRICHLET, 0.0, None),
            right_bc=(BC_DIRICHLET, 4.0, None),
            screened=True,
        )
        prob, _ = assemble_problem(mesh)
        # All nodes strictly in/on [1, 2] must share exactly ONE row index!
        metal_tets = np.flatnonzero(mesh.element_materials == 1)
        metal_nodes = np.unique(mesh.elements[metal_tets].ravel())
        self.assertEqual(np.unique(prob.row[metal_nodes]).size, 1)
        self.assertNotEqual(int(prob.row[metal_nodes[0]]), NONE_U32)

        res_direct, _ = solve_problem(prob, method="direct")
        field, _ = build_nodal_field(mesh, prob, res_direct)
        # Exact piecewise linear solution -> machine precision!
        x = mesh.nodes[:, 0]
        u_exact = np.where(x <= 1.0, 3.0 * x, np.where(x <= 2.0, 3.0, 3.0 + (x - 2.0)))
        np.testing.assert_allclose(field.u, u_exact, rtol=1e-12, atol=1e-12)

        if HAS_PYAMG:
            res_amg, sinfo = solve_problem(prob, method="amg", rtol=1e-12)
            self.assertEqual(sinfo.warnings, [])
            self.assertLess(sinfo.residual, 1e-11)
            np.testing.assert_allclose(res_amg.x, res_direct.x, rtol=1e-10, atol=1e-10)

    def test_two_disconnected_floating_metals_same_material(self) -> None:
        # [0, 5]: diel [0,1], metal [1,2], diel [2,3], metal [3,4], diel [4,5]; all eps=1, u(0)=0, u(5)=3
        # Exact potentials: metal1 = 1.0 V, metal2 = 2.0 V!
        materials = [
            ("diel", 1.0, 0.0, False, None),
            ("copper", 1.0, 0.0, True, None),
        ]

        def mat_fn(cx: float, _cy: float, _cz: float) -> int:
            return 1 if (1.0 < cx < 2.0 or 3.0 < cx < 4.0) else 0

        mesh = _build_box_mesh(
            5, 2, 2,
            lx=5.0,
            material_fn=mat_fn,
            materials=materials,
            left_bc=(BC_DIRICHLET, 0.0, None),
            right_bc=(BC_DIRICHLET, 3.0, None),
            screened=False,
        )
        prob, _ = assemble_problem(mesh)
        res, _ = solve_problem(prob, method="direct")
        field, _ = build_nodal_field(mesh, prob, res)

        m1_nodes = np.flatnonzero((mesh.nodes[:, 0] >= 1.0 - 1e-9) & (mesh.nodes[:, 0] <= 2.0 + 1e-9))
        m2_nodes = np.flatnonzero((mesh.nodes[:, 0] >= 3.0 - 1e-9) & (mesh.nodes[:, 0] <= 4.0 + 1e-9))
        self.assertEqual(np.unique(prob.row[m1_nodes]).size, 1)
        self.assertEqual(np.unique(prob.row[m2_nodes]).size, 1)
        self.assertNotEqual(int(prob.row[m1_nodes[0]]), int(prob.row[m2_nodes[0]]))
        np.testing.assert_allclose(field.u[m1_nodes], 1.0, atol=1e-12)
        np.testing.assert_allclose(field.u[m2_nodes], 2.0, atol=1e-12)

    def test_floating_metal_touching_dirichlet_and_short_circuit_error(self) -> None:
        # Floating metal in [0, 1] touches left Dirichlet wall 2.5V -> whole metal becomes known at 2.5V!
        materials = [
            ("metal", 1.0, 0.0, True, None),
            ("diel", 1.0, 0.0, False, None),
        ]
        mesh = _build_box_mesh(
            4, 2, 2,
            lx=2.0,
            material_fn=lambda cx, _cy, _cz: 0 if cx < 1.0 else 1,
            materials=materials,
            left_bc=(BC_DIRICHLET, 2.5, None),
            right_bc=(BC_DIRICHLET, 4.5, None),
            screened=False,
        )
        prob, _ = assemble_problem(mesh)
        metal_nodes = np.unique(mesh.elements[mesh.element_materials == 0].ravel())
        self.assertTrue(np.all(prob.row[metal_nodes] == NONE_U32))
        res, _ = solve_problem(prob, method="direct")
        field, _ = build_nodal_field(mesh, prob, res)
        x = mesh.nodes[:, 0]
        u_exact = np.where(x <= 1.0, 2.5, 2.5 + 2.0 * (x - 1.0))
        np.testing.assert_allclose(field.u, u_exact, atol=1e-12)

        # Prescribed metal at 5.0V touching Dirichlet wall at 2.5V -> short-circuit error!
        bad_materials = [
            ("metal", 1.0, 0.0, True, 5.0),
            ("diel", 1.0, 0.0, False, None),
        ]
        bad_mesh = _build_box_mesh(
            2, 1, 1,
            lx=2.0,
            material_fn=lambda cx, _cy, _cz: 0 if cx < 1.0 else 1,
            materials=bad_materials,
            left_bc=(BC_DIRICHLET, 2.5, None),
            right_bc=(BC_DIRICHLET, 4.5, None),
            screened=False,
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            assemble_problem(bad_mesh)

    def test_error_conditions_charge_in_metal_and_interior_boundary(self) -> None:
        # 1. Trapped charge inside a metal tet -> error
        mesh_metal_charge = _build_box_mesh(
            2, 1, 1,
            lx=2.0,
            material_fn=lambda cx, _cy, _cz: 0 if cx < 1.0 else 1,
            materials=[("metal", 1.0, 0.0, True, 0.0), ("diel", 1.0, 0.0, False, None)],
            charges=[((0.5, 0.5, 0.5), -1)],
            screened=False,
        )
        with self.assertRaisesRegex(ValueError, "metal"):
            assemble_problem(mesh_metal_charge)

        # 2. Interior boundary element (face shared by two tets) -> error
        base = _build_box_mesh(2, 1, 1, lx=2.0, screened=False)
        # Find an interior face (shared by two tets)
        face_count: dict[tuple[int, int, int], int] = {}
        for tet in base.elements:
            for k in range(4):
                f = tuple(sorted(int(tet[j]) for j in range(4) if j != k))
                face_count[f] = face_count.get(f, 0) + 1
        interior_face = next(f for f, c in face_count.items() if c == 2)
        bad_be = np.vstack([base.boundary_elements, np.asarray(interior_face, dtype=np.uint32)])
        bad_be_bcs = np.concatenate([base.boundary_element_bcs, np.asarray([0], dtype=np.uint16)])
        mesh_int_bc = MeshData(
            nodes=base.nodes,
            elements=base.elements,
            element_materials=base.element_materials,
            boundary_elements=bad_be,
            boundary_element_bcs=bad_be_bcs,
            bc_type=base.bc_type,
            bc_value=base.bc_value,
            bc_has_alpha=base.bc_has_alpha,
            bc_alpha=base.bc_alpha,
            material_names=base.material_names,
            material_eps_r=base.material_eps_r,
            material_screening_length=base.material_screening_length,
            material_is_metal=base.material_is_metal,
            material_has_voltage=base.material_has_voltage,
            material_voltage=base.material_voltage,
            charge_positions=base.charge_positions,
            charge_values=base.charge_values,
            neighbours=base.neighbours,
            screened=base.screened,
        )
        with self.assertRaisesRegex(ValueError, "interior face"):
            assemble_problem(mesh_int_bc)

    def test_tiny_permittivity_dirichlet_box_is_not_singular(self) -> None:
        # The singularity check is structural: scaling eps_r by 1e-14 must not trip it.
        mesh = _build_box_mesh(4, 2, 2, lx=2.0, materials=[("tiny", 1e-14, 0.0, False, None)], screened=False)
        prob, info = assemble_problem(mesh)
        self.assertEqual(info.warnings, [])
        self.assertLess(float(np.max(np.abs(prob.A.data))), 1e-13)
        methods = ["direct", "amg"] if HAS_PYAMG else ["direct"]
        for method in methods:
            res, sinfo = solve_problem(prob, method=method)
            self.assertLess(sinfo.residual, 1e-12)
            self.assertEqual(sinfo.warnings, [])
            field, _ = build_nodal_field(mesh, prob, res)
            np.testing.assert_allclose(field.u, 0.5 * mesh.nodes[:, 0], atol=1e-11)

        # Same box with only Neumann ends is singular whatever the scale.
        floating = _build_box_mesh(
            2, 1, 1, materials=[("tiny", 1e-14, 0.0, False, None)],
            left_bc=(BC_NEUMANN, 0.0, None), right_bc=(BC_NEUMANN, 1e-14, None), screened=False,
        )
        with self.assertRaisesRegex(ValueError, "singular"):
            assemble_problem(floating)
        # A Robin alpha > 0 face pins it, even a tiny one.
        robin_pinned = _build_box_mesh(
            2, 1, 1, materials=[("tiny", 1e-14, 0.0, False, None)],
            left_bc=(BC_ROBIN, 0.0, 1e-14), right_bc=(BC_NEUMANN, 1e-14, None), screened=False,
        )
        assemble_problem(robin_pinned)

    def test_supported_range_pipeline_never_rejects_its_own_output(self) -> None:
        # Extreme mesh inputs inside [1e-30, 1e30]: assemble -> write/read -> solve ->
        # write/read -> field -> write/read accepts every intermediate output.
        for eps_r, volts, length in ((1e30, 1e30, 1.0), (1e-30, 1e-30, 1.0), (1e30, 1e30, 1e30),
                                     (1e-30, 1e-30, 1e-29), (1e30, -1e30, 1e-29), (1e-30, 1e30, 1e30)):
            with self.subTest(eps_r=eps_r, volts=volts, length=length):
                mesh = _build_box_mesh(
                    4, 2, 2, lx=length, ly=length, lz=length, materials=[("m", eps_r, 0.0, False, None)],
                    right_bc=(BC_DIRICHLET, volts, None), screened=False,
                )
                tag = f"{eps_r:g}_{volts:g}_{length:g}"
                mesh_h5 = self.tmp / f"range_mesh_{tag}.h5"
                write_mesh_properties(mesh_h5, mesh)
                prob, _ = assemble_problem(read_mesh_properties(mesh_h5))
                ap_h5 = self.tmp / f"range_ap_{tag}.h5"
                write_assembled_problem(ap_h5, prob)
                res, sinfo = solve_problem(read_assembled_problem(ap_h5), method="direct")
                self.assertLess(sinfo.residual, 1e-12)
                sr_h5 = self.tmp / f"range_sr_{tag}.h5"
                write_solver_result(sr_h5, res)
                field, _ = build_nodal_field(mesh, read_assembled_problem(ap_h5), read_solver_result(sr_h5))
                nf_h5 = self.tmp / f"range_nf_{tag}.h5"
                write_nodal_field(nf_h5, field)
                np.testing.assert_allclose(read_nodal_field(nf_h5).u, volts * mesh.nodes[:, 0] / length,
                                           rtol=1e-10, atol=1e-10 * abs(volts))

        # Rounding-noise coordinates next to zero (as refined meshes produce) are accepted.
        noisy = _build_box_mesh(2, 1, 1, screened=False)
        noisy_nodes = noisy.nodes.copy()
        noisy_nodes[noisy_nodes == 0.0] = -7.5e-32
        assemble_problem(_replace_mesh(noisy, nodes=noisy_nodes))

        # Sol's case: eps_r = 1e95 (old range) used to yield f ~ 1e110 that solve rejected; it is
        # now refused up front as a mesh input.
        with self.assertRaisesRegex(ValueError, r"material_eps_r .*\[1e-30, 1e30\]"):
            assemble_problem(_build_box_mesh(2, 1, 1, materials=[("m", 1e95, 0.0, False, None)],
                                             right_bc=(BC_DIRICHLET, 1e15, None), screened=False))
        # Products of supported inputs can still leave the AssembledProblem range: assemble then
        # refuses instead of writing a problem that read_assembled_problem / solve reject.
        big = _build_box_mesh(2, 1, 1, lx=1e30, ly=1e30, lz=1e30, screened=False)
        with self.assertRaisesRegex(ValueError, r"assembled f .*\[1e-100, 1e100\]"):
            assemble_problem(big, constant_source=1e30)          # ~ 1e30 * h^3 ~ 1e118
        screened = _build_box_mesh(2, 1, 1, lx=1e10, ly=1e10, lz=1e10,
                                   materials=[("m", 1e30, 1e-30, False, None)], screened=True)
        with self.assertRaisesRegex(ValueError, r"assembled A\.data .*\[1e-100, 1e100\]"):
            assemble_problem(screened)                           # eps_r / sl^2 * h^3 ~ 1e118

        # Sol's case: A = 1e-75, f = 1e75 (both in range) solve to x = 1e150; SolverResult only
        # has to be finite, so read_solver_result and the field stage accept it.
        mesh = _build_box_mesh(1, 1, 1, screened=False)
        n_nodes = mesh.nodes.shape[0]
        row = np.full(n_nodes, NONE_U32, dtype=np.uint32)
        row[0] = 0
        prob = AssembledProblemData(
            A=sparse.csc_matrix(np.array([[1e-75]])), f=np.array([1e75]), row=row,
            known_node=np.arange(1, n_nodes, dtype=np.uint32), known_value=np.zeros(n_nodes - 1),
        )
        ap_h5 = self.tmp / "huge_x_ap.h5"
        write_assembled_problem(ap_h5, prob)
        res, _ = solve_problem(read_assembled_problem(ap_h5), method="direct")
        self.assertEqual(float(res.x[0]), 1e150)
        sr_h5 = self.tmp / "huge_x_sr.h5"
        write_solver_result(sr_h5, res)
        field, _ = build_nodal_field(mesh, read_assembled_problem(ap_h5), read_solver_result(sr_h5))
        nf_h5 = self.tmp / "huge_x_nf.h5"
        write_nodal_field(nf_h5, field)
        self.assertEqual(float(read_nodal_field(nf_h5).u[0]), 1e150)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_nodal_field(mesh, prob, SolverResultData(x=np.array([np.inf])))

    def test_cli_subcommands_end_to_end(self) -> None:
        mesh = _build_box_mesh(3, 2, 2, lx=1.0, screened=False)
        mesh_h5 = self.tmp / "mesh.h5"
        prob_h5 = self.tmp / "prob.h5"
        prob_info = self.tmp / "prob.info.json"
        sol_h5 = self.tmp / "sol.h5"
        sol_info = self.tmp / "sol.info.json"
        fld_h5 = self.tmp / "fld.h5"
        fld_info = self.tmp / "fld.info.json"
        write_mesh_properties(mesh_h5, mesh)

        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "python")}
        subprocess.run(
            [sys.executable, "-m", "poisson_test_solver", "assemble",
             "--mesh", str(mesh_h5), "--out", str(prob_h5), "--info-out", str(prob_info)],
            check=True,
            env=env,
        )
        subprocess.run(
            [sys.executable, "-m", "poisson_test_solver", "solve",
             "--in", str(prob_h5), "--method", "direct", "--out", str(sol_h5), "--info-out", str(sol_info)],
            check=True,
            env=env,
        )
        subprocess.run(
            [sys.executable, "-m", "poisson_test_solver", "field",
             "--mesh", str(mesh_h5), "--problem", str(prob_h5), "--solution", str(sol_h5),
             "--out", str(fld_h5), "--info-out", str(fld_info)],
            check=True,
            env=env,
        )

        p_info = json.loads(prob_info.read_text(encoding="utf-8"))
        s_info = json.loads(sol_info.read_text(encoding="utf-8"))
        self.assertEqual(p_info["type"], "AssemblyResult::Info")
        self.assertIn("nnz", p_info)
        self.assertIn("n_dirichlet", p_info)
        self.assertEqual(s_info["type"], "SolverResult::Info")
        self.assertEqual(s_info["iterations"], 1)
        self.assertLess(s_info["residual"], 1e-13)

        fld = read_nodal_field(fld_h5)
        np.testing.assert_allclose(fld.u, mesh.nodes[:, 0], atol=1e-12)


@unittest.skipUnless(HAS_CORE_DEPS, CORE_DEPS_SKIP_REASON)
@unittest.skipUnless(HAS_SKFEM, "scikit-fem not installed")
class TestStagesIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stages_ignore_pythonpath_for_the_package(self) -> None:
        mesh_h5 = _box_mesh_h5(self.tmp)
        # A shadow package that would break the stage if the child imported it.
        shadow = self.tmp / "shadow" / "poisson_test_solver"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("raise ImportError('shadow copy imported')\n", encoding="utf-8")

        def pipeline(run: RunContext) -> dict:
            prob = stage_assemble(run, mesh_h5)
            sol = stage_solve(run, prob, method="direct")
            fld = stage_build_field(run, mesh_h5, prob, sol)
            return {"field": fld}

        env_without = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for label, env in (
            ("unset", env_without),
            ("shadowed", {**env_without, "PYTHONPATH": str(shadow.parent)}),
        ):
            with self.subTest(pythonpath=label), mock.patch.dict(os.environ, env, clear=True):
                report = run_experiment(pipeline, cache=Cache(self.tmp / f"cache_{label}"))
                fld = read_nodal_field(report.branches[0].outputs["field"].path)
                mesh = read_mesh_properties(mesh_h5)
                np.testing.assert_allclose(fld.u, mesh.nodes[:, 0], atol=1e-12)

        # The bootstrap refuses a package directory other than the one it was told to load.
        stage = make_solve_stage()
        from poisson_runner.stages import StageCall

        call = StageCall(executable=stage.executable, inputs={}, outputs={"out": self.tmp / "x.h5"},
                         info_path=self.tmp / "x.json", settings={}, config_path=self.tmp / "c.json")
        cmd = stage.build_command(call)
        self.assertEqual(cmd[1:3], ["-P", "-c"])
        self.assertEqual(Path(cmd[4]), Path(ts_cli.__file__).resolve().parent)
        self.assertEqual(cmd[5], "solve")
        cmd[4] = str(shadow)
        res = subprocess.run(cmd, capture_output=True, text=True, env=env_without)
        self.assertNotEqual(res.returncode, 0)

    def test_system_python_and_venv_get_distinct_code_hashes(self) -> None:
        if sys.prefix == sys.base_prefix:
            self.skipTest("tests are not running inside a virtualenv")
        venv_py = Path(os.path.abspath(sys.executable))
        system_py = Path("/usr/bin/python3") if Path("/usr/bin/python3").exists() else venv_py.resolve()
        cache = Cache(self.tmp / "hash_cache")
        s_venv = make_solve_stage(python_executable=venv_py)
        s_sys = make_solve_stage(python_executable=system_py)
        self.assertEqual(s_venv.executable, venv_py)   # symlink kept unresolved
        self.assertIn(f"prefix={sys.prefix}", s_venv.version)
        self.assertNotEqual(s_venv.version, s_sys.version)
        self.assertNotEqual(s_venv.compute_code_hash(cache), s_sys.compute_code_hash(cache))

    @unittest.skipUnless(HAS_PYAMG, "pyamg not installed")
    def test_amg_solve_is_deterministic_across_repeats_and_cold_runs(self) -> None:
        mesh_h5 = _box_mesh_h5(self.tmp, nx=8, ny=4, nz=4, side_bc=(BC_ROBIN, 0.0, 0.3))

        def pipeline(run: RunContext) -> dict:
            prob = stage_assemble(run, mesh_h5)
            sol = stage_solve(run, prob, method="amg", repeats=2)
            return {"solution": sol}

        outputs = []
        for k in range(2):
            report = run_experiment(pipeline, cache=Cache(self.tmp / f"amg_cache_{k}"))
            sol = report.branches[0].outputs["solution"]
            self.assertFalse(sol.cached)
            self.assertEqual(sol.execution.get("nondeterministic_outputs"), [])
            self.assertFalse(sol.execution.get("warnings"))
            self.assertEqual(sol.info["warnings"], [])
            self.assertGreater(sol.info["iterations"], 1)
            outputs.append(Path(sol.path).read_bytes())
        self.assertEqual(outputs[0], outputs[1])


@unittest.skipUnless(HAS_CORE_DEPS, CORE_DEPS_SKIP_REASON)
@unittest.skipUnless(HAS_SKFEM and HAS_GMSH, "requires scikit-fem and gmsh")
class TestRunnerVerificationAndAMR(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_planar_layers_p1_patch_test_is_exact(self) -> None:
        # Unscreened two-dielectric + floating metal + Robin end: the exact solution is piecewise
        # linear in x, so P1 on the interface-conforming Gmsh mesh must reproduce it at every node.
        # u = D x / eps_a on [0, t_a]; metal at V_m = D t_a / eps_a; u = V_m + D (x - x_b) / eps_b
        # after it; the Robin end eps_b u' + alpha u = g gives D (1 + alpha (t_a/eps_a + t_b/eps_b)) = g.
        t_a, t_m, t_b = 1.0, 0.5, 1.0
        eps_a, eps_b, alpha, g = 2.0, 5.0, 0.5, 2.0
        case = Case.of(
            "planar_layers",
            layers=[
                # Screening lengths are present but screened=False must drop them.
                {"name": "diel_a", "thickness": t_a, "eps_r": eps_a, "screening_length": 1.5},
                {"name": "floating_m", "thickness": t_m, "metal": True},
                {"name": "diel_b", "thickness": t_b, "eps_r": eps_b, "screening_length": 2.0},
            ],
            left=dirichlet(0.0),
            right=robin(alpha, g),
            width=1.0,
            height=1.0,
            screened=False,
        )
        D = g / (1.0 + alpha * (t_a / eps_a + t_b / eps_b))
        v_m = D * t_a / eps_a
        x_b = t_a + t_m
        methods = ["direct", "amg"] if HAS_PYAMG else ["direct"]

        def pipeline(run: RunContext) -> dict:
            h = compare("h_max", [0.5, 0.25])
            method = compare("method", methods)
            mesh = run.case_mesh(case, h_max=h)
            sol = solve_mesh(run, mesh, method=method, rtol=1e-14)
            return {"mesh": mesh, "field": sol["field"]}

        report = run_experiment(pipeline, cache=Cache(self.tmp / "patch_cache"))
        self.assertEqual(len(report.branches), 2 * len(methods))
        for br in report.branches:
            mesh = read_mesh_properties(br.outputs["mesh"].path)
            self.assertFalse(mesh.screened)
            x = mesh.nodes[:, 0]
            u_exact = np.where(x <= t_a, D * x / eps_a, np.where(x <= x_b, v_m, v_m + D * (x - x_b) / eps_b))
            u = read_nodal_field(br.outputs["field"].path).u
            err = float(np.max(np.abs(u - u_exact)))
            self.assertLess(err, 1e-11, f"{br.choice_labels}: max nodal error {err:.3e}")

    def test_planar_layers_convergence_and_amg_agreement(self) -> None:
        # Screened two-dielectric + floating metal + Robin end verification case!
        case = Case.of(
            "planar_layers",
            layers=[
                {"name": "diel_a", "thickness": 1.0, "eps_r": 2.0, "screening_length": 1.5},
                {"name": "floating_m", "thickness": 0.5, "metal": True},
                {"name": "diel_b", "thickness": 1.0, "eps_r": 5.0, "screening_length": 2.0},
            ],
            left=dirichlet(0.0),
            right=robin(0.5, 2.0),
            width=1.0,
            height=1.0,
            screened=True,
        )
        cache = Cache(self.tmp / "conv_cache")
        methods = ["direct", "amg"] if HAS_PYAMG else ["direct"]

        def pipeline(run: RunContext) -> dict:
            h = compare("h_max", [0.5, 0.25])
            method = compare("method", methods)
            mesh = run.case_mesh(case, h_max=h)
            prob = stage_assemble(run, mesh)
            # repeats=2 re-runs the solve and checks the output bytes match (AMG is seeded).
            sol = stage_solve(run, prob, method=method, rtol=1e-12, repeats=2)
            fld = stage_build_field(run, mesh, prob, sol)
            return {"mesh": mesh, "problem": prob, "solution": sol, "field": fld}

        report = run_experiment(pipeline, cache=cache)
        self.assertEqual(len(report.branches), 2 * len(methods))
        for br in report.branches:
            self.assertEqual(br.outputs["solution"].execution.get("nondeterministic_outputs"), [])
            self.assertEqual(br.outputs["solution"].info["warnings"], [])

        # Verify direct vs amg match tightly on the exact same mesh (including floating metal!)
        if HAS_PYAMG:
            for h_lbl in ("0.5", "0.25"):
                b_dir = [b for b in report.branches if b.choice_labels == {"h_max": h_lbl, "method": "direct"}][0]
                b_amg = [b for b in report.branches if b.choice_labels == {"h_max": h_lbl, "method": "amg"}][0]
                u_dir = read_nodal_field(b_dir.outputs["field"].path).u
                u_amg = read_nodal_field(b_amg.outputs["field"].path).u
                np.testing.assert_allclose(u_amg, u_dir, rtol=1e-10, atol=1e-10)

        # Verify floating metal has constant potential and mesh refinement moves solution smoothly
        b_coarse = [b for b in report.branches if b.choice_labels == {"h_max": "0.5", "method": "direct"}][0]
        b_fine = [b for b in report.branches if b.choice_labels == {"h_max": "0.25", "method": "direct"}][0]
        m_fine = read_mesh_properties(b_fine.outputs["mesh"].path)
        u_fine = read_nodal_field(b_fine.outputs["field"].path).u
        metal_mat = m_fine.material_names.index("floating_m")
        metal_nodes = np.unique(m_fine.elements[m_fine.element_materials == metal_mat].ravel())
        self.assertLess(float(np.ptp(u_fine[metal_nodes])), 1e-14)
        self.assertGreater(float(np.mean(u_fine[metal_nodes])), 0.0)
        self.assertGreater(b_fine.outputs["problem"].info["nnz"], b_coarse.outputs["problem"].info["nnz"])

    def test_closed_loop_amr_with_dry_run_and_point_charge(self) -> None:
        # Screened half-space image case with a point charge -> exercises solve_mesh -> poisson_adapt
        # with exclude_patch across multiple cycles and dry-run discovery!
        case = Case.of(
            "grounded_ball_image",
            radius=5.0,
            eps_r=2.0,
            charge=-1,
            position=[1.5, 0.0, 0.0],
        )
        cache = Cache(self.tmp / "amr_cache")

        max_cycles = 3

        def amr_pipeline(run: RunContext) -> dict:
            est = compare("estimator", ["kelly", "recovery"])
            mesh = run.case_mesh(case, h_max=2.0)
            state = None
            sol = None
            cycles_run = 0
            cfg = {
                "estimator": {"type": est, "point_sources": "exclude_patch"},
                "marker": {"type": "dorfler", "theta": 0.5},
                "refiner": {"type": "red_green"},
                "adapter": {"max_elements": 50000},
            }
            # PoissonDriver's cycle order: solve, stop after the last solve, otherwise adapt.
            for cycle in range(max_cycles):
                sol = solve_mesh(run, mesh, method="direct")
                cycles_run += 1
                if cycle + 1 == max_cycles:
                    break
                res = run.adapt_mesh(mesh, sol["field"], state=state, config=cfg)
                mesh, state = res["mesh"], res["state"]
                if not run.dry_run and mesh.info["status"] != "adapted":
                    break
            assert sol is not None
            return {"mesh": mesh, "field": sol["field"], "cycles": cycles_run}

        # Dry run must succeed cleanly without reading placeholder files or breaking on status.
        dry = run_experiment(amr_pipeline, cache=cache, dry_run=True)
        self.assertEqual(len(dry.branches), 2)

        # Real multi-cycle AMR run!
        report = run_experiment(amr_pipeline, cache=cache)
        self.assertEqual(len(report.branches), 2)
        for br in report.branches:
            self.assertEqual(br.outputs["cycles"], max_cycles)
            self.assertEqual(br.outputs["mesh"].info["status"], "adapted")
            self.assertGreater(
                br.outputs["mesh"].info["elements_after"],
                br.outputs["mesh"].info["elements_before"],
            )
            # The returned field was solved on the returned (last adapted) mesh.
            mesh = read_mesh_properties(br.outputs["mesh"].path)
            fld = read_nodal_field(br.outputs["field"].path)
            self.assertEqual(len(fld.u), len(mesh.nodes))
            self.assertEqual(fld.point_source_position.shape, (1, 3))
            self.assertNotEqual(int(fld.point_source_element[0]), NONE_U32)
            self.assertLess(int(fld.point_source_element[0]), len(mesh.elements))

        # Second identical run must be 100% cache hits (proving determinism + caching)!
        report2 = run_experiment(amr_pipeline, cache=cache)
        self.assertEqual(report2.cache_misses, 0)
        self.assertGreater(report2.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
