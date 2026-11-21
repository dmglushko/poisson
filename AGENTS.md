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
