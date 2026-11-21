"""Content-addressed caching and loop-free Compare() runner for Poisson stages."""

from .cache import Artifact, Cache, CacheLease, CacheMode, CacheStats, normalize_cache_mode
from .driver import BranchResult, Compare, ExperimentReport, RunContext, StageTrace, compare, run_experiment
from .executor import ExecutionRequest, ExecutionResult, Executor, LocalExecutor
from .hashing import blake3_bytes, blake3_file, blake3_json, blake3_version
from .stages import (
    Stage,
    StageCall,
    make_adapt_stage,
    make_import_free_charges_stage,
    make_import_mesh_stage,
    resolve_binary,
)

__all__ = [
    "Artifact",
    "BranchResult",
    "Cache",
    "CacheLease",
    "CacheMode",
    "CacheStats",
    "Compare",
    "ExecutionRequest",
    "ExecutionResult",
    "Executor",
    "ExperimentReport",
    "LocalExecutor",
    "RunContext",
    "Stage",
    "StageCall",
    "StageTrace",
    "blake3_bytes",
    "blake3_file",
    "blake3_json",
    "blake3_version",
    "compare",
    "make_adapt_stage",
    "make_import_free_charges_stage",
    "make_import_mesh_stage",
    "normalize_cache_mode",
    "resolve_binary",
    "run_experiment",
]
