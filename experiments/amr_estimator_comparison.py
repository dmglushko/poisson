"""Experiment: compare all AMR error estimators on analytical verification cases.

Runs one screened and one unscreened variant for each verification case that supports
screening, plus the unscreened-only grounded_ball_image case, using Red-Green refinement
and Dorfler marking across all four C++ error estimators:
  - kelly
  - recovery
  - residual
  - trajectory

Usage:
    PYTHONPATH=python ~/.venvs/poisson/bin/python experiments/amr_estimator_comparison.py
"""

from __future__ import annotations

from pathlib import Path

from poisson_runner import Cache, Case, RunContext, compare, dirichlet, run_experiment
from poisson_test_solver.stages import solve_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".poisson_cache"
RESULTS_DIR = REPO_ROOT / "results" / "amr_estimator_comparison"

MAX_CYCLES = 5          # cycles 0, 1, 2, 3, 4 (initial solve + 4 refinement steps)
MAX_ELEMENTS = 25_000   # safety cap so every branch stays fast locally

# Each entry defines:
#   - case: the C++ VerificationCase specification
#   - gmsh: initial coarse mesh settings (coarse enough that AMR has room to work)
CASES: dict[str, dict] = {
    "planar_screened": {
        "case": Case.of(
            "planar_layers",
            layers=[{"name": "oxide", "thickness": 4.0, "eps_r": 3.9, "screening_length": 1.0}],
            left=dirichlet(0.0),
            right=dirichlet(1.0),
            width=2.0,
            height=2.0,
            screened=True,
        ),
        "gmsh": {"h_max": 1.0},
    },
    "planar_unscreened": {
        "case": Case.of(
            "planar_layers",
            layers=[{"name": "oxide", "thickness": 4.0, "eps_r": 3.9, "screening_length": 1.0}],
            left=dirichlet(0.0),
            right=dirichlet(1.0),
            width=2.0,
            height=2.0,
            screened=False,
        ),
        "gmsh": {"h_max": 1.0},
    },
    "spherical_screened": {
        "case": Case.of(
            "spherical_layers",
            inner_radius=1.0,
            layers=[{"name": "oxide", "thickness": 3.0, "eps_r": 3.9, "screening_length": 1.5}],
            inner=dirichlet(1.0),
            outer=dirichlet(0.0),
            screened=True,
        ),
        "gmsh": {"h_max": 1.2},
    },
    "spherical_unscreened": {
        "case": Case.of(
            "spherical_layers",
            inner_radius=1.0,
            layers=[{"name": "oxide", "thickness": 3.0, "eps_r": 3.9, "screening_length": 1.5}],
            inner=dirichlet(1.0),
            outer=dirichlet(0.0),
            screened=False,
        ),
        "gmsh": {"h_max": 1.2},
    },
    "grounded_ball_unscreened": {
        "case": Case.of(
            "grounded_ball_image",
            radius=5.0,
            eps_r=1.0,
            charge=-1,
            position=[1.5, 0.0, 0.0],
        ),
        "gmsh": {"h_max": 2.0, "h_charge": 0.6, "r_fine": 0.8, "r_coarse": 2.5},
    },
    "half_space_screened": {
        "case": Case.of(
            "half_space_image",
            eps_r=1.0,
            screening_length=2.0,
            charge=-1,
            distance=2.0,
            half_width=8.0,
            height=8.0,
            plane="dirichlet",
            screened=True,
        ),
        "gmsh": {"h_max": 3.0, "h_charge": 0.8, "r_fine": 1.0, "r_coarse": 4.0},
    },
    "half_space_unscreened": {
        "case": Case.of(
            "half_space_image",
            eps_r=1.0,
            screening_length=2.0,
            charge=-1,
            distance=2.0,
            half_width=8.0,
            height=8.0,
            plane="dirichlet",
            screened=False,
        ),
        "gmsh": {"h_max": 3.0, "h_charge": 0.8, "r_fine": 1.0, "r_coarse": 4.0},
    },
    "dielectric_interface_screened": {
        "case": Case.of(
            "dielectric_interface",
            eps_upper=1.0,
            eps_lower=4.0,
            screening_length=2.0,
            charge=-1,
            distance=2.0,
            half_width=8.0,
            half_height=8.0,
            screened=True,
        ),
        "gmsh": {"h_max": 3.5, "h_charge": 0.8, "r_fine": 1.0, "r_coarse": 4.0},
    },
    "dielectric_interface_unscreened": {
        "case": Case.of(
            "dielectric_interface",
            eps_upper=1.0,
            eps_lower=4.0,
            screening_length=2.0,
            charge=-1,
            distance=2.0,
            half_width=8.0,
            half_height=8.0,
            screened=False,
        ),
        "gmsh": {"h_max": 3.5, "h_charge": 0.8, "r_fine": 1.0, "r_coarse": 4.0},
    },
}

ESTIMATORS = ["kelly", "recovery", "residual", "trajectory"]


def pipeline(run: RunContext) -> dict:
    # 1. Declare the two sweep axes.
    #    Every combination (9 cases x 4 estimators = 36 branches) is explored automatically.
    case_spec = compare("case", CASES)
    estimator = compare("estimator", ESTIMATORS)

    # 2. Build the initial coarse mesh for this case (cached and shared across all 4 estimators!).
    case: Case = case_spec["case"]
    mesh = run.case_mesh(case, **case_spec["gmsh"])

    # 3. Configure AMR: Red-Green refiner + Dorfler marking + chosen estimator.
    #    `residual` and `trajectory` require an explicit smooth RHS (`constant_source: 0.0`
    #    here, since all verification cases have f = 0 outside point-charge deltas).
    amr_config = {
        "constant_source": 0.0,
        "estimator": {"type": estimator, "point_sources": "exclude_patch"},
        "marker": {"type": "dorfler", "theta": 0.5},
        "refiner": {"type": "red_green"},
        "adapter": {"max_elements": MAX_ELEMENTS},
    }

    # 4. Run the adaptive loop: solve -> record -> adapt -> repeat.
    state = None
    history: list[dict] = []
    sol = None

    for cycle in range(MAX_CYCLES):
        sol = solve_mesh(run, mesh, method="auto", direct_threshold=15_000)

        entry: dict = {
            "cycle": cycle,
            "mesh": mesh,
            "field": sol["field"],
        }

        if cycle + 1 == MAX_CYCLES:
            history.append(entry)
            break

        adapted = run.adapt_mesh(mesh, sol["field"], state=state, config=amr_config)
        # Store the estimator's total indicator on this mesh before moving to the refined mesh.
        entry["adapt_info"] = adapted["mesh"].info
        history.append(entry)

        mesh = adapted["mesh"]
        state = adapted["state"]

        # Only inspect runtime status during a real run (not dry-run discovery).
        if not run.dry_run and mesh.info.get("status") != "adapted":
            break

    assert sol is not None
    return {
        "case": case,
        "cycles_run": len(history),
        "history": history,
        "mesh": mesh,
        "field": sol["field"],
    }


def main() -> None:
    cache = Cache(CACHE_DIR)
    print(f"Running AMR estimator comparison ({len(CASES)} cases x {len(ESTIMATORS)} estimators)...")
    report = run_experiment(pipeline, cache=cache)
    print(report.format_table())

    failed = [b for b in report.branches if not b.ok]
    if failed:
        print(f"\nWARNING: {len(failed)} branch(es) failed:")
        for b in failed:
            print(f"  branch #{b.index} {b.choice_labels}: {b.error}")

    saved_path = report.save(RESULTS_DIR, overwrite=True)
    print(f"\nSaved run to: {saved_path}")


if __name__ == "__main__":
    main()
