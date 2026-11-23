"""Smallest real experiment, for checking a DelftBlue setup end to end.

    bash scripts/delftblue/submit.sh scripts/delftblue/smoke_test.py --time 00:05:00

Imports one tiny mesh with three materials: three branches, one piece each, then a collect job.
"""

from pathlib import Path

from poisson_runner import RunContext, compare, experiment_main

HERE = Path(__file__).resolve().parent / "smoke"


def pipeline(run: RunContext) -> dict:
    material = compare("epsilon_r", {"2.0": "low", "7.5": "mid", "11.7": "high"})
    mesh = run.import_mesh(HERE / "box.msh", materials=[HERE / f"siliconnitride_{material}.yaml"])
    return {"mesh": mesh}


if __name__ == "__main__":
    raise SystemExit(experiment_main(pipeline))
