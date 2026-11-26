# `poisson_test_solver` Architecture

**Scope:** External-library Python test solver for closed-loop AMR testing and independent
verification of future C++ `Assembly` and `Solver` stages. It is **not** a production C++
pipeline stage and is never linked into the C++ library or `PoissonDriver`.

## Purpose & Split

The package mirrors the project's HDF5 stage contracts 1-to-1 as three CLI subcommands so that
when a teammate's C++ `Assembly` or `Solver` arrives, either stage can be swapped in or compared
side by side via `poisson_runner`:

```text
MeshProperties (.h5)
  ──▶ python -m poisson_test_solver assemble ──▶ AssembledProblem (.h5)
  ──▶ python -m poisson_test_solver solve    ──▶ SolverResult (.h5)
  ──▶ python -m poisson_test_solver field    ──▶ NodalField (.h5)
  ──▶ poisson_adapt                          ──▶ refined MeshProperties (.h5)
```

## Module Layout

| Module | Responsibility |
|---|---|
| `constants.py` | CODATA 2022 constants matching `include/common/constants.h` (`E_OVER_EPS0_NM`, `COULOMB_CONSTANT_NM`) |
| `h5_io.py` | Deterministic HDF5 I/O (`libver=('v108','v108')`, `track_times=False`, sorted int32 CSC `A`, atomic rename) + JSON sidecars matching `AssemblyResult::Info` and `SolverResult::Info`. Readers validate strictly once at the boundary, mirroring the C++ readers in `src/io/` (exact on-disk types and ranks, scalar `uint32`/`uint64` attributes, `{0, 1}` flags, index bounds, sorted CSC) and additionally reject non-finite floats |
| `validation.py` | Strict scalar checks shared by the API and the CLI JSON config (`require_int` rejects `true` and `2.9`; `require_finite_float` rejects `nan`/`inf`) and the supported-magnitude check (see below) |
| `locate.py` | Vectorized AABB + signed-distance point-charge locator (safe on graded AMR meshes; each candidate tet is shifted by a vertex and normalized by its extent, so huge or tiny coordinates neither overflow nor underflow) with deterministic tie-breaking |
| `assemble.py` | `scikit-fem` P1 tetrahedral assembly + exact `row`/`known` floating-metal reduction ($A = P^T K_{\text{full}} P$) |
| `solve.py` | `scipy.sparse.linalg.spsolve` (`direct`) and `pyamg` smoothed-aggregation V-cycle + `scipy.sparse.linalg.cg` (`amg`) |
| `field.py` | Reconstructs nodal `u` from `AssembledProblem` + `SolverResult` and attaches located `point_sources` for `poisson_adapt` |
| `compare.py` | Permutation-invariant `AssembledProblem` comparator (canonicalizes rows by minimum mesh node index per row; symmetric, row-scaled tolerances, see below) |
| `cli.py`, `__main__.py` | `python -m poisson_test_solver assemble\|solve\|field` with input/output path-aliasing checks |
| `stages.py` | `poisson_runner` `Stage` factories and helpers (`assemble`, `solve`, `build_field`, `solve_mesh`), kept here rather than in `poisson_runner/stages.py` |

## Proposed Assembly Semantics (pending agreement with Assembly owner)

These conventions govern `assemble.py` until `src/assembly/architecture.md` defines the
production C++ contract:

1. **PDE & screening:** On non-metal tetrahedra, $-\nabla \cdot (\varepsilon_r \nabla u) + c u = f$
   with $c = \varepsilon_r / \lambda^2$ when `mesh.screened && screening_length > 0`, and $c = 0$
   otherwise. Metal tetrahedra are skipped during volume assembly.
2. **Metals (`row` and `known`):**
   - Metal nodes are grouped into connected components through shared vertices of metal tets.
   - If a component has a prescribed voltage (`material.voltage`) or touches a Dirichlet boundary
     node, every node in that component becomes `known` at that voltage (conflicting voltages are
     an error).
   - Every remaining floating-metal component gets **one shared row index** in `row`. Reducing
     with the prolongation matrix $P$ ($A = P^T K_{\text{full}} P$,
     $f = P^T (b_{\text{full}} - K_{\text{full}} u_{\text{known}})$) symmetrically eliminates known
     nodes and enforces zero net flux (Gauss's law) on each floating metal without penalty
     conditioning issues.
   - `known` is sorted by ascending node index; free rows are ordered deterministically.
3. **Boundary elements:** Every boundary triangle must be an outer face of one tetrahedron
   (interior faces shared by two tets are rejected, matching the importer). Unlisted outer faces
   are natural Neumann ($g = 0$). Robin/Neumann faces use $\varepsilon_r \partial_n u + \alpha u = g$
   ($+\int_\Gamma \alpha u v$ on the LHS, $+\int_\Gamma g v$ on the RHS). A Neumann $g = 0$ face
   on a metal outer boundary is a no-op; non-zero Neumann or non-trivial Robin on a metal outer
   face is an error.
4. **Trapped charges:** Deposited as P1 barycentric delta loads ($q \cdot e / (\varepsilon_0\,\text{nm})$).
   Tet $K$ admits a charge at $p$ iff $\min_i d_i \ge -\delta$, where $d_i$ is the signed
   distance (in nm) from $p$ to the face opposite vertex $i$ ($d_i = \lambda_i \cdot 3|K| / |F_i|$)
   and $\delta_K = 64\,\varepsilon_{\text{mach}} (\|p\|_\infty + \text{extent}(K))$. "Inside metal"
   is judged by the metal tets' own (small) tolerances, so the large $\delta$ of a huge neighbouring
   dielectric tet cannot swallow a charge sitting in the metal. A charge admitted by metal tets is
   inside the metal iff some admitted metal tet holds it strictly ($\min_i d_i > \delta_K$), or every
   admitted metal tet keeps it off its metal–dielectric interface faces
   ($\min_{i:\,\text{nbr}_i\ \text{dielectric}} d_i > \delta_K$, $+\infty$ without such a face — e.g. on
   a metal–metal face). Interface faces come from `MeshProperties::neighbours`. A charge outside the
   mesh, inside the metal, or admitted only by metal tets is an error. Among admitting tets the pick
   prefers dielectric tetrahedra, then the lowest tetrahedron index, so a charge on a
   metal–dielectric interface (face, edge, or vertex) lands in the dielectric. `.cha` positions are
   float32 ($\sim 10^{-7}\|p\|$), so placement decisions finer than that are below input resolution.
5. **Singularity check (structural):** A free row is *pinned* if it has a nonzero `K_full`
   coupling to a known node, touches a Robin face with $\alpha > 0$, or touches a non-metal tet
   with $c > 0$. Every connected component of $A$ must contain a pinned row, otherwise assembly
   fails. Because the test is structural it is scale-invariant (an $\varepsilon_r = 10^{-14}$ box
   pinned by Dirichlet is fine). The energy ratio $v_C^T A v_C / \sum_{i \in C} |A_{ii}|$ of the
   constant indicator $v_C$ is only an ill-conditioning warning (when $\le 10^{-12}$).

## Supported Value Range

Every floating-point value must be finite. Nonzero magnitudes are bounded per data kind:

| Data | Nonzero range |
|---|---|
| Mesh / assembly inputs: `eps_r`, `screening_length`, material `voltage`, BC `value`/`alpha`, `constant_source`, `source(...)` output | $10^{-30} \le \lvert v \rvert \le 10^{30}$ |
| Coordinates: `nodes`, charge positions | $\lvert v \rvert \le 10^{30}$ (no lower bound) |
| `AssembledProblem`: `A`, `f`, `known_value` | $10^{-100} \le \lvert v \rvert \le 10^{100}$ |
| `SolverResult::x`, `NodalField::u` | any finite float64 |

Readers and `assemble_problem`/`solve_problem` (for in-memory inputs) raise `ValueError` naming
the field and the range. Coordinates have no lower bound because only their differences (edge
vectors) enter the assembly, and refined meshes legitimately hold rounding-noise coordinates such as
$-7.5 \cdot 10^{-32}$ next to zero. Products or ratios of two or three mesh inputs (`eps_r * V / L`,
$(\varepsilon_r / \lambda) / \lambda$) then stay well inside the `AssembledProblem` range, and
within that range the solver's RHS normalization and residual stay far from float64 overflow and the
subnormal range. `assemble_problem` also checks its own `A` and `f` against the `AssembledProblem`
range (extreme combinations such as a huge volume times a huge source can leave it), so
`assemble` never writes a problem that `read_assembled_problem` or `solve` rejects. `x` only has to
be finite: `field` copies it into `u` without arithmetic, so any `x` that `solve` returns is valid
for `field`. Trapped-charge positions in `.cha` are float32.

## Comparator Tolerances

`compare_assembled_problems(first, second, rtol=1e-10, atol=0.0)` is symmetric in its arguments
and scaled row by row, so one large row cannot hide a wrong small row and scaling the whole
system does not change the verdict. With $s_i = \max(\|A_1[i,:]\|_\infty, \|A_2[i,:]\|_\infty)$
and $U = \max |u_{\text{known}}|$ over both problems:

- $A$ (union of both patterns): $|A_1[i,j] - A_2[i,j]| \le \text{atol} + \text{rtol}\, s_i$
- $f$: $|f_1[i] - f_2[i]| \le \text{atol} + \text{rtol} \max(|f_1[i]|, |f_2[i]|, s_i U)$
  ($s_i U$ bounds the Dirichlet lifting $-K u_{\text{known}}$ inside $f$)
- known values: $|v_1[k] - v_2[k]| \le \text{atol} + \text{rtol} \max(|v_1[k]|, |v_2[k]|)$

When $U > 0$ the $f$ test is evaluated divided by $U$
($|\Delta f_i| / U \le \text{atol}/U + \text{rtol}\max(|f_1[i]|/U, |f_2[i]|/U, s_i)$), so $s_i U$
never overflows into a tolerance of $\infty$. Non-finite values in `A`, `f`, or `known_value` make
the problems unequal.

`reason` names the worst offending canonical row (with its smallest mesh node) or known node.

## Solver Tiers & Dependencies

- **Dependencies:** `numpy`, `scipy`, `h5py` (for I/O, direct solve, field build, compare),
  `scikit-fem` (for `assemble`), and `pyamg` (for `solve --method amg`).
- **RHS normalization:** Both methods solve $A y = f / \max|f|$ and return $x = \max|f| \cdot y$,
  so right-hand sides like $10^{-90}$ or $10^{90}$ neither underflow nor overflow. The reported
  `residual` is the true relative residual of the **returned** $x$ in the normalized system,
  $\|A (x / \max|f|) - b\|_2 / \|b\|_2$; a warning is recorded iff it exceeds `rtol`. A zero `f`
  returns `x = 0` with `residual = 0`; a non-finite or out-of-range `f` or `A` is an error.
- **`--method direct`:** `scipy.sparse.linalg.spsolve` (SuperLU). Exact to machine precision;
  intended for tests and small/medium 3D meshes (practical ceiling ~100k–200k DOFs).
- **`--method amg`:** `pyamg` smoothed-aggregation V-cycle preconditioner + `scipy.sparse.linalg.cg`
  (default `rtol = 1e-12`). A warning is recorded iff the final true relative residual exceeds
  `rtol` (not from CG's exit code). PyAMG draws random vectors for its spectral-radius estimates,
  so the hierarchy is built under `np.random.seed(seed)` (`--seed`, config `"seed"`, default 0)
  and the caller's global RNG state is restored afterwards; outputs are byte-for-byte reproducible.
  A module lock serializes that save–seed–build–restore section between concurrent
  `solve_problem` calls.
  Scales to much larger meshes; if multi-million-DOF runs on DelftBlue later bottleneck on
  single-threaded PyAMG, `solve.py` alone can add a `petsc4py`/Hypre backend without touching any
  other stage.
- **`--method auto`:** Uses `direct` up to `direct_threshold` (default 100,000 unknowns) and `amg`
  above it.

## `poisson_runner` Integration & AMR Loop Pattern

`stages.py` tracks all package `.py` files in `code_files`. It asks the child interpreter itself
(one small subprocess per stage build, run with `os.environ` overlaid by the stage's `env`, the
same environment the runner launches it with) for its `sys.prefix`, `sys.version`, and the
installed versions of the packages each stage imports, and embeds them in `Stage.version`. The
probe is not memoized, so `PYTHONPATH`/`PYTHONHOME`/`PYTHONNOUSERSITE`/`PYTHONUSERBASE` changes
are always reflected; the helpers key `run.get_stage(...)` on the interpreter, the per-stage
`env` (`assemble(run, mesh, env={...})`), and the effective values of those four variables
(`os.environ` overlaid by `env`), so each distinct `env`, and an inherited `PYTHONPATH` change
within one experiment, gets its own freshly probed Stage. Two interpreters that are the same binary (e.g. `/usr/bin/python3` and a venv
`bin/python` symlinked to it) therefore get different code hashes, and library upgrades
invalidate cached entries.

Stages launch `python -P -c <bootstrap> <package dir> <subcommand> ...` rather than
`python -m poisson_test_solver`: the bootstrap puts the hashed package's parent directory first
on `sys.path`, checks that `poisson_test_solver` was imported from exactly that directory, and
calls `cli.main`. The stage does not depend on `PYTHONPATH` to find the package and cannot load a
different copy than the one hashed. `-P` keeps the working directory off `sys.path`; `-I`/`-E`
are not used, so `PYTHONPATH` and user site-packages still provide the dependencies (DelftBlue).
The package directory is an argument, not part of the hashed command, so the cache key does not
depend on where the checkout lives.

When writing a multi-cycle AMR loop in `poisson_runner`, follow `PoissonDriver`'s cycle order
(solve, stop after the last solve, otherwise adapt) and keep dry-run discovery and SLURM sharding
safe: put all `compare(...)` calls before the loop, iterate for a fixed `max_cycles`, and only
inspect `status` to exit early when `not run.dry_run`. The returned `mesh` is then always the mesh
the returned `solution` was computed on:

```python
from poisson_runner import compare, run_experiment
from poisson_test_solver.stages import solve_mesh

def pipeline(run):
    est = compare("estimator", ["kelly", "recovery"])
    mesh = run.case_mesh(case, h_max=0.5)
    state = None
    for cycle in range(max_cycles):
        sol = solve_mesh(run, mesh)
        if cycle + 1 == max_cycles:
            break
        res = run.adapt_mesh(mesh, sol["field"], state=state, config=...)
        mesh, state = res["mesh"], res["state"]
        if not run.dry_run and mesh.info["status"] != "adapted":
            break
    return {"mesh": mesh, "solution": sol["field"]}
```
