# Project Conventions for Agents

## Architecture Documentation
- Root [architecture.md](architecture.md) defines **only** the high-level pipeline data flow,
  shared data class definitions, and short stage contracts (with a link to each stage's or
  subsystem's own `architecture.md`).
- Do **not** add stage or subsystem implementation details (such as input parsing rules,
  solver/AMR internals, CLI layouts, serialization internals, or caching mechanics) to root
  `architecture.md`.
- Instead, document each stage or subsystem's detailed design in an `architecture.md` file
  inside its own folder (for example, [src/importer/architecture.md](src/importer/architecture.md),
  `src/amr/architecture.md`, `cli/architecture.md`, or
  `python/poisson_runner/architecture.md`).

## Stage Isolation
- Keep the core C++ library (`include/`, `src/`) free of caching and orchestration concerns.
- Keep stage interfaces centered on the shared data classes defined in root
  [architecture.md](architecture.md).
- New pipeline stages should also provide a thin CLI under `cli/<stage>/` following
  [cli/architecture.md](cli/architecture.md).

## Core C++ Design Conventions
These are the preferred patterns across the codebase (follow them by default unless a stage has a clear reason to diverge):
- **Composition & drivers:** Prefer composing drivers and composite stages out of small, interchangeable module interfaces (typically injected via `std::shared_ptr<const T>`).
- **`Settings` structs:** When a module or driver needs configuration, the preferred convention is a nested `struct Settings` (with sensible defaults) passed to and validated in its constructor (`std::invalid_argument`). Drivers keep their own general `Settings` (e.g. budgets, stopping criteria) separate from their sub-modules' settings.
- **Results & `Info`:** Pure domain structs (`MeshProperties`, `FreeCharges`) hold only problem data. Stage and module outputs carry a nested `Info` with at least `double time` and `std::vector<std::string> warnings` (and optionally typed metrics, a `details` map, or `StageInfo` for composites). Keep deterministic output data separate from `Info` so CLIs can write `.h5` data and `.info.json` metadata independently.
- **Non-Virtual Interface (NVI):** When a polymorphic interface has shared input validation, timing, or post-processing across implementations, prefer a non-virtual public entrypoint that handles those shared steps and delegates the algorithm to a private `virtual` hook.
- **Immutability & sessions:** Configured modules are generally `const` and reusable across runs; when a run needs cross-cycle state, hold it in a short-lived `Session` (e.g. `MeshAdapter::start()`).
- **Errors vs. warnings:** Throw `std::invalid_argument` for invalid API arguments or settings, `std::runtime_error` for bad external input or unrecoverable runtime failures, and record suspicious-but-usable conditions in `info.warnings`.
- **Reference examples:** For idiomatic examples, see [include/amr/error_estimator.h](include/amr/error_estimator.h) (interface, `Settings`, `Info`) and [include/amr/error_based_adapter.h](include/amr/error_based_adapter.h) / [include/amr/mesh_adapter.h](include/amr/mesh_adapter.h) (composition, driver `Settings`, `Session`).
