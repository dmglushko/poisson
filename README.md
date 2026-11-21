# Screened Poisson field solver for Nebula

A finite-element library for the screened Poisson equation on tetrahedral meshes, with ray
tracing through the resulting field. It is meant to replace deal.II in the Nebula
electron-matter simulator.

The high-level data flow, data classes, and stage contracts are described in
[architecture.md](architecture.md). So far the data classes, the Importer
([src/importer/architecture.md](src/importer/architecture.md)), shared geometry/quadrature
primitives ([src/common/architecture.md](src/common/architecture.md)), adaptive mesh
refinement ([src/amr/architecture.md](src/amr/architecture.md)), stage CLIs & deterministic
HDF5 serialization ([cli/architecture.md](cli/architecture.md)), and the Python caching/compare
runner ([python/poisson_runner/architecture.md](python/poisson_runner/architecture.md)) are
implemented.

## Build and test

Needs a C++23 compiler, CMake, Eigen3 (`libeigen3-dev` on Ubuntu), the HDF5 library
(`libhdf5-dev`), and Python 3 (`python3-h5py` for inspecting `.h5` files in tests). BLAKE3
v1.8.7 is vendored under `third_party/blake3/` so no external hashing dependency is required.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Stage CLI and caching runner

Each implemented pipeline stage has a thin C++ CLI binary under `cli/<stage>/`
(`cli/importer/poisson_import.cpp` → `build/poisson_import`,
`cli/amr/poisson_adapt.cpp` → `build/poisson_adapt`) that writes deterministic HDF5 data
files and separate JSON `Info` sidecars:

```bash
./build/poisson_import --mesh box.msh --material silicon.yaml --out mesh.h5 --info-out mesh.info.json
./build/poisson_import --primaries beam.pri --out free_charges.h5 --info-out free_charges.info.json
./build/poisson_adapt  --mesh mesh.h5 --solution field.h5 --config amr.json --out adapted.h5 --info-out adapt.info.json
```

Use `poisson_runner` in Python to run loop-free `compare(...)` experiments with automatic
BLAKE3 content-addressed caching, binary-hash invalidation, 4-mode cache control
(`"use"`, `"read_only"`, `"refresh"`, `"off"`), repeated timing statistics, and `dry_run=True`
previews:

```python
from poisson_runner import Cache, compare, run_experiment

def experiment(run):
    mesh = run.import_mesh("box.msh", materials=["silicon.yaml"])
    # Add downstream Stage(...) calls with compare(...) without writing loops
    return mesh

report = run_experiment(
    experiment,
    cache=Cache(".poisson_cache", max_bytes=20 * 1024**3),
)
print(report.format_table())
```

Manage the scratch cache from the command line:

```bash
PYTHONPATH=python python3 -m poisson_runner cache status
PYTHONPATH=python python3 -m poisson_runner cache inspect
PYTHONPATH=python python3 -m poisson_runner cache clean --max-gb 10
PYTHONPATH=python python3 -m poisson_runner cache clean --all
```
