"""Command-line interface: `python -m poisson_test_solver assemble|solve|field`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .assemble import assemble_problem
from .field import build_nodal_field
from .h5_io import (
    read_assembled_problem,
    read_mesh_properties,
    read_solver_result,
    write_assembled_problem,
    write_assembly_info_json,
    write_field_info_json,
    write_nodal_field,
    write_solver_info_json,
    write_solver_result,
)
from .solve import DEFAULT_DIRECT_THRESHOLD, solve_problem
from .validation import require_finite_float
from .validation import require_int as _require_int  # rejects `true`, `2.9`, strings in JSON config


def _paths_alias(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return a.resolve() == b.resolve()


def _check_aliasing(inputs: Sequence[Path], outputs: Sequence[tuple[str, Path]]) -> None:
    for i, (flag_i, out_i) in enumerate(outputs):
        if out_i.is_dir():
            raise ValueError(f"{flag_i} points to an existing directory: {out_i}")
        for inp in inputs:
            if _paths_alias(out_i, inp):
                raise ValueError(f"{flag_i} aliases input file '{inp}'")
        for flag_j, out_j in outputs[i + 1 :]:
            if _paths_alias(out_i, out_j):
                raise ValueError(f"{flag_i} aliases {flag_j}")


def _load_config(config_path: Path | None, allowed_keys: set[str], section: str) -> dict[str, Any]:
    if config_path is None:
        return {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: config root must be a JSON object")
    unknown = sorted(set(raw) - allowed_keys)
    if unknown:
        raise ValueError(f"{section}: unknown config key(s): {', '.join(unknown)}")
    return raw


def _cmd_assemble(args: argparse.Namespace) -> int:
    mesh_path = Path(args.mesh)
    out_path = Path(args.out)
    config_path = Path(args.config) if args.config else None
    info_path = Path(args.info_out) if args.info_out else None

    inputs = [mesh_path] + ([config_path] if config_path else [])
    outputs: list[tuple[str, Path]] = [("--out", out_path)]
    if info_path is not None:
        outputs.append(("--info-out", info_path))
    _check_aliasing(inputs, outputs)

    cfg = _load_config(config_path, {"constant_source", "intorder"}, "assemble")
    constant_source = cfg.get("constant_source", args.constant_source)
    if constant_source is not None:
        constant_source = require_finite_float(constant_source, "constant_source")
    intorder = _require_int(cfg.get("intorder", args.intorder), "intorder")

    mesh = read_mesh_properties(mesh_path)
    problem, info = assemble_problem(mesh, constant_source=constant_source, intorder=intorder)
    write_assembled_problem(out_path, problem)
    if info_path is not None:
        write_assembly_info_json(info_path, info)
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    in_path = Path(args.problem_in)
    out_path = Path(args.out)
    config_path = Path(args.config) if args.config else None
    info_path = Path(args.info_out) if args.info_out else None

    inputs = [in_path] + ([config_path] if config_path else [])
    outputs: list[tuple[str, Path]] = [("--out", out_path)]
    if info_path is not None:
        outputs.append(("--info-out", info_path))
    _check_aliasing(inputs, outputs)

    cfg = _load_config(config_path, {"method", "rtol", "maxiter", "direct_threshold", "seed"}, "solve")
    method = cfg.get("method", args.method)
    if not isinstance(method, str):
        raise ValueError(f"method must be a string, got {method!r}")
    rtol = require_finite_float(cfg.get("rtol", args.rtol), "rtol")
    maxiter = _require_int(cfg.get("maxiter", args.maxiter), "maxiter")
    direct_threshold = _require_int(cfg.get("direct_threshold", args.direct_threshold), "direct_threshold")
    seed = _require_int(cfg.get("seed", args.seed), "seed")

    problem = read_assembled_problem(in_path)
    result, info = solve_problem(
        problem,
        method=method,
        rtol=rtol,
        maxiter=maxiter,
        direct_threshold=direct_threshold,
        seed=seed,
    )
    write_solver_result(out_path, result)
    if info_path is not None:
        write_solver_info_json(info_path, info)
    return 0


def _cmd_field(args: argparse.Namespace) -> int:
    mesh_path = Path(args.mesh)
    problem_path = Path(args.problem)
    solution_path = Path(args.solution)
    out_path = Path(args.out)
    info_path = Path(args.info_out) if args.info_out else None

    inputs = [mesh_path, problem_path, solution_path]
    outputs: list[tuple[str, Path]] = [("--out", out_path)]
    if info_path is not None:
        outputs.append(("--info-out", info_path))
    _check_aliasing(inputs, outputs)

    mesh = read_mesh_properties(mesh_path)
    problem = read_assembled_problem(problem_path)
    solution = read_solver_result(solution_path)
    field_data, info = build_nodal_field(mesh, problem, solution)
    write_nodal_field(out_path, field_data)
    if info_path is not None:
        write_field_info_json(info_path, info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m poisson_test_solver",
        description="External-library Python test solver (verification & AMR testing only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_asm = sub.add_parser("assemble", help="Assemble MeshProperties .h5 -> AssembledProblem .h5")
    p_asm.add_argument("--mesh", required=True, help="Input MeshProperties .h5")
    p_asm.add_argument("--config", help="Optional JSON config (constant_source, intorder)")
    p_asm.add_argument("--constant-source", type=float, default=None, help="Optional constant smooth RHS f [V/nm^2]")
    p_asm.add_argument("--intorder", type=int, default=2, help="Volume quadrature order (default: 2)")
    p_asm.add_argument("--out", required=True, help="Output AssembledProblem .h5")
    p_asm.add_argument("--info-out", help="Optional AssemblyResult::Info .json")
    p_asm.set_defaults(func=_cmd_assemble)

    p_sol = sub.add_parser("solve", help="Solve AssembledProblem .h5 -> SolverResult .h5")
    p_sol.add_argument("--in", dest="problem_in", required=True, help="Input AssembledProblem .h5")
    p_sol.add_argument("--config", help="Optional JSON config (method, rtol, maxiter, direct_threshold, seed)")
    p_sol.add_argument("--method", choices=("auto", "direct", "amg"), default="auto", help="Solver method")
    p_sol.add_argument("--rtol", type=float, default=1e-12, help="Relative tolerance for AMG-CG")
    p_sol.add_argument("--maxiter", type=int, default=2000, help="Max CG iterations for AMG-CG")
    p_sol.add_argument(
        "--direct-threshold",
        type=int,
        default=DEFAULT_DIRECT_THRESHOLD,
        help="DOF cutoff for method=auto",
    )
    p_sol.add_argument("--seed", type=int, default=0, help="NumPy RNG seed for the PyAMG setup (default: 0)")
    p_sol.add_argument("--out", required=True, help="Output SolverResult .h5")
    p_sol.add_argument("--info-out", help="Optional SolverResult::Info .json")
    p_sol.set_defaults(func=_cmd_solve)

    p_fld = sub.add_parser("field", help="Build NodalField .h5 for poisson_adapt")
    p_fld.add_argument("--mesh", required=True, help="Input MeshProperties .h5")
    p_fld.add_argument("--problem", required=True, help="Input AssembledProblem .h5")
    p_fld.add_argument("--solution", required=True, help="Input SolverResult .h5")
    p_fld.add_argument("--out", required=True, help="Output NodalField .h5")
    p_fld.add_argument("--info-out", help="Optional NodalFieldBuild::Info .json")
    p_fld.set_defaults(func=_cmd_field)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
