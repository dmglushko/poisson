# Screened Poisson field solver for Nebula

A finite-element library for the screened Poisson equation on tetrahedral meshes, with ray
tracing through the resulting field. It is meant to replace deal.II in the Nebula
electron-matter simulator.

The design is described in [architecture.md](architecture.md): the data flow, the data
classes and the steps. So far the data classes, the Importer and adaptive mesh refinement
([src/amr/architecture.md](src/amr/architecture.md)) are implemented.

## Build and test

Needs a C++23 compiler, CMake, Eigen3 (`libeigen3-dev` on Ubuntu) and the HDF5 library
(`libhdf5-dev`).

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
