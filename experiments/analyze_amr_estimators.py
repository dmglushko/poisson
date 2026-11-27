"""Analysis script: evaluate the saved AMR estimator comparison run against exact solutions.

Demonstrates the post-hoc analysis pattern in `poisson_runner`:
  1. `load_run(...)` reloads the saved experiment (without re-running any solves or AMR).
  2. `evaluate(saved, analyze_branch)` runs our analysis over every successful branch.
  3. Inside `analyze_branch`, we read each cycle's cached MeshProperties and NodalField `.h5`
     files, integrate the exact L2 error using 4-point tetrahedral Gauss quadrature, and
     collect the estimator's own error indicator (`eta_total`) from `adapt_info`.
  4. Saves the evaluation report, a per-cycle CSV, and an interactive HTML chart with toggles.

Usage:
    PYTHONPATH=python ~/.venvs/poisson/bin/python experiments/analyze_amr_estimators.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poisson_runner import BranchResult, Case, RunContext, evaluate, load_run
from poisson_test_solver.constants import COULOMB_CONSTANT_NM
from poisson_test_solver.h5_io import MeshData, read_mesh_properties, read_nodal_field

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "amr_estimator_comparison"
ANALYSIS_DIR = REPO_ROOT / "results" / "amr_estimator_comparison_analysis"

# 4-point symmetric Gauss quadrature on the reference tetrahedron (exact for degree <= 2).
_A = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0
_B = (5.0 - math.sqrt(5.0)) / 20.0
_TET_QUAD_BARY = np.array(
    [
        [_A, _B, _B, _B],
        [_B, _A, _B, _B],
        [_B, _B, _A, _B],
        [_B, _B, _B, _A],
    ],
    dtype=np.float64,
)


def _yukawa(r: np.ndarray, k: float) -> np.ndarray:
    return np.exp(-k * r) / r if k > 0.0 else 1.0 / r


def exact_u_for_case(case: Case, x: np.ndarray, mat_names: list[str], tet_mat: np.ndarray) -> np.ndarray:
    """Evaluate the exact analytical potential u(x) matching C++ `src/verification/`.

    x: (E, Q, 3) quadrature point coordinates in nm.
    tet_mat: (E,) material index per tetrahedron.
    Returns: (E, Q) float64 array of exact potentials in V.
    """
    p = case.params
    name = case.name
    screened = case.screened

    if name == "planar_layers":
        layer = p["layers"][0]
        L = float(layer["thickness"])
        lam = float(layer.get("screening_length", 0.0))
        k = (1.0 / lam) if (screened and lam > 0.0) else 0.0
        coord_x = x[..., 0]
        if k == 0.0:
            return coord_x / L
        return np.sinh(k * coord_x) / np.sinh(k * L)

    if name == "spherical_layers":
        r_in = float(p["inner_radius"])
        layer = p["layers"][0]
        r_out = r_in + float(layer["thickness"])
        lam = float(layer.get("screening_length", 0.0))
        k = (1.0 / lam) if (screened and lam > 0.0) else 0.0
        r = np.clip(np.linalg.norm(x, axis=-1), r_in, r_out)
        if k == 0.0:
            return (r_in / (r_out - r_in)) * (r_out / r - 1.0)
        return (r_in * np.sinh(k * (r_out - r))) / (r * np.sinh(k * (r_out - r_in)))

    if name == "grounded_ball_image":
        R = float(p["radius"])
        eps_r = float(p["eps_r"])
        q = float(p["charge"])
        # Match C++ float32 rounding of .cha charge positions.
        pos = np.array(p["position"], dtype=np.float32).astype(np.float64)
        d = float(np.linalg.norm(pos))
        q_img = -q * R / d
        pos_img = pos * (R * R / (d * d))
        r1 = np.linalg.norm(x - pos, axis=-1)
        r2 = np.linalg.norm(x - pos_img, axis=-1)
        return (COULOMB_CONSTANT_NM / eps_r) * (q / r1 + q_img / r2)

    if name == "half_space_image":
        eps_r = float(p["eps_r"])
        lam = float(p["screening_length"])
        k = (1.0 / lam) if (screened and lam > 0.0) else 0.0
        q = float(p["charge"])
        d = float(np.float32(p["distance"]))
        sign = -1.0 if p.get("plane", "dirichlet") == "dirichlet" else 1.0
        r_plus = np.linalg.norm(x - np.array([0.0, 0.0, d]), axis=-1)
        r_minus = np.linalg.norm(x - np.array([0.0, 0.0, -d]), axis=-1)
        return (q * COULOMB_CONSTANT_NM / eps_r) * (_yukawa(r_plus, k) + sign * _yukawa(r_minus, k))

    if name == "dielectric_interface":
        e1 = float(p["eps_upper"])
        e2 = float(p["eps_lower"])
        rho = (e1 - e2) / (e1 + e2)
        lam = float(p["screening_length"])
        k = (1.0 / lam) if (screened and lam > 0.0) else 0.0
        q = float(p["charge"])
        d = float(np.float32(p["distance"]))
        r_plus = np.linalg.norm(x - np.array([0.0, 0.0, d]), axis=-1)
        r_minus = np.linalg.norm(x - np.array([0.0, 0.0, -d]), axis=-1)
        g_plus = _yukawa(r_plus, k)
        g_minus = _yukawa(r_minus, k)

        upper_idx = mat_names.index("upper")
        is_upper = (tet_mat == upper_idx)[:, None]
        u_upper = (q * COULOMB_CONSTANT_NM / e1) * (g_plus + rho * g_minus)
        u_lower = (2.0 * q * COULOMB_CONSTANT_NM / (e1 + e2)) * g_plus
        return np.where(is_upper, u_upper, u_lower)

    raise ValueError(f"Unsupported verification case: {name}")


def exclusion_info(case: Case) -> tuple[np.ndarray | None, float]:
    """Return (charge_position, exclusion_radius) for point-charge cases, else (None, 0)."""
    p = case.params
    if case.name == "grounded_ball_image":
        pos = np.array(p["position"], dtype=np.float32).astype(np.float64)
        r_excl = min(1.0, 0.5 * (float(p["radius"]) - float(np.linalg.norm(pos))))
        return pos, r_excl
    if case.name == "half_space_image":
        d = float(np.float32(p["distance"]))
        r_excl = min(1.0, 0.5 * d, 0.5 * float(p["half_width"]), 0.5 * (float(p["height"]) - d))
        return np.array([0.0, 0.0, d]), r_excl
    if case.name == "dielectric_interface":
        d = float(np.float32(p["distance"]))
        r_excl = min(1.0, 0.5 * d, 0.5 * float(p["half_width"]), 0.5 * (float(p["half_height"]) - d))
        return np.array([0.0, 0.0, d]), r_excl
    return None, 0.0


def compute_l2_error(case: Case, mesh: MeshData, u_nodal: np.ndarray) -> dict[str, float]:
    """Compute absolute and relative L2 error on `mesh` using 4-point tet Gauss quadrature."""
    # Vertices of every tet: shape (E, 4, 3)
    verts = mesh.nodes[mesh.elements]
    # Tet volumes: |det(v1-v0, v2-v0, v3-v0)| / 6
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    e3 = verts[:, 3] - verts[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", e1, np.cross(e2, e3))) / 6.0

    # Quadrature points per tet: (E, 4, 3)
    q_pts = np.einsum("qi,eik->eqk", _TET_QUAD_BARY, verts)

    # Exclude tetrahedra inside the exclusion ball around a point charge singularity.
    charge_pos, r_excl = exclusion_info(case)
    if charge_pos is not None and r_excl > 0.0:
        min_dist = np.min(np.linalg.norm(q_pts - charge_pos, axis=-1), axis=-1)
        keep = min_dist >= r_excl
    else:
        keep = np.ones(len(mesh.elements), dtype=bool)

    q_pts = q_pts[keep]
    vol_keep = vol[keep]
    tet_mat = mesh.element_materials[keep]

    # Interpolate P1 FEM solution to quadrature points: (E_keep, 4)
    u_tet = u_nodal[mesh.elements[keep]]
    u_h = np.einsum("qi,ei->eq", _TET_QUAD_BARY, u_tet)
    u_ex = exact_u_for_case(case, q_pts, mesh.material_names, tet_mat)

    diff_sq = np.mean((u_h - u_ex) ** 2, axis=-1)
    ex_sq = np.mean(u_ex**2, axis=-1)

    l2_abs = float(np.sqrt(np.sum(vol_keep * diff_sq)))
    l2_norm = float(np.sqrt(np.sum(vol_keep * ex_sq)))
    l2_rel = l2_abs / l2_norm if l2_norm > 0.0 else float("nan")
    return {"l2_abs": l2_abs, "l2_rel": l2_rel}


def extract_eta_total(adapt_info: dict[str, Any] | None) -> float | None:
    """Pull the estimator's total error indicator from `AdaptationResult::Info`."""
    if not isinstance(adapt_info, dict):
        return None
    for stage in adapt_info.get("stages", []):
        if stage.get("stage") == "estimate":
            total = stage.get("metrics", {}).get("total")
            if isinstance(total, (int, float)):
                return float(total)
    return None


def build_interactive_html(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write a self-contained interactive HTML chart with toggles for cases, estimators, and scale."""
    data_json = json.dumps(records)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AMR Estimator Convergence</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    font-family: var(--font-sans, system-ui, -apple-system, sans-serif);
    color: var(--foreground, #e5e7eb);
  }}
  .wrap {{
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    background: var(--card, rgba(255,255,255,0.04));
    border: 1px solid var(--border, rgba(255,255,255,0.12));
    border-radius: var(--radius, 8px);
    padding: 10px 12px;
  }}
  .group {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }}
  .group-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted-foreground, #9ca3af);
    margin-right: 2px;
  }}
  select {{
    background: var(--secondary, rgba(255,255,255,0.08));
    color: var(--foreground, #e5e7eb);
    border: 1px solid var(--border, rgba(255,255,255,0.18));
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
  }}
  .pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid var(--border, rgba(255,255,255,0.18));
    background: var(--secondary, rgba(255,255,255,0.06));
    color: var(--foreground, #e5e7eb);
    cursor: pointer;
    user-select: none;
    transition: opacity 0.12s ease, border-color 0.12s ease;
  }}
  .pill.off {{
    opacity: 0.38;
    text-decoration: line-through;
  }}
  .swatch {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
  }}
  .divider {{
    width: 1px;
    height: 18px;
    background: var(--border, rgba(255,255,255,0.14));
    margin: 0 2px;
  }}
  .chart-card {{
    position: relative;
    background: var(--card, rgba(255,255,255,0.03));
    border: 1px solid var(--border, rgba(255,255,255,0.12));
    border-radius: var(--radius, 8px);
    padding: 12px 14px 10px 14px;
  }}
  .chart-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .chart-title {{
    font-size: 14px;
    font-weight: 600;
  }}
  .chart-sub {{
    font-size: 11px;
    color: var(--muted-foreground, #9ca3af);
  }}
  svg {{
    width: 100%;
    height: 360px;
    display: block;
    overflow: visible;
  }}
  .grid-line {{
    stroke: var(--border, rgba(255,255,255,0.12));
    stroke-width: 1;
  }}
  .axis-label {{
    fill: var(--muted-foreground, #9ca3af);
    font-size: 11px;
    font-family: var(--font-mono, ui-monospace, monospace);
  }}
  .axis-title {{
    fill: var(--muted-foreground, #9ca3af);
    font-size: 12px;
    font-weight: 500;
  }}
  .legend-note {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 11px;
    color: var(--muted-foreground, #9ca3af);
    margin-top: 6px;
  }}
  .tooltip {{
    position: absolute;
    pointer-events: none;
    display: none;
    background: var(--popover, #18181b);
    color: var(--popover-foreground, #f4f4f5);
    border: 1px solid var(--border, rgba(255,255,255,0.2));
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 11px;
    font-family: var(--font-mono, ui-monospace, monospace);
    line-height: 1.45;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    z-index: 10;
    white-space: nowrap;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="group">
      <span class="group-label">Case</span>
      <select id="caseSelect"></select>
    </div>
    <div class="divider"></div>
    <div class="group" id="estToggles">
      <span class="group-label">Estimators</span>
    </div>
  </div>

  <div class="toolbar">
    <div class="group">
      <span class="group-label">Series</span>
      <button class="pill" id="toggleReal" type="button">Solid: Real L2 Error</button>
      <button class="pill" id="toggleEta" type="button">Dashed: Estimated Error (eta)</button>
    </div>
    <div class="divider"></div>
    <div class="group">
      <span class="group-label">Metric</span>
      <select id="metricSelect">
        <option value="l2_rel" selected>Relative L2</option>
        <option value="l2_abs">Absolute L2 (V*nm^1.5)</option>
      </select>
    </div>
    <div class="group">
      <span class="group-label">X</span>
      <select id="xSelect">
        <option value="cycle" selected>Iteration (cycle)</option>
        <option value="n_elements">Elements (log)</option>
      </select>
    </div>
    <div class="group">
      <span class="group-label">Y</span>
      <select id="yScaleSelect">
        <option value="log" selected>Log</option>
        <option value="linear">Linear</option>
      </select>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-head">
      <div class="chart-title" id="chartTitle"></div>
      <div class="chart-sub" id="chartSub"></div>
    </div>
    <svg id="svg" viewBox="0 0 700 360" preserveAspectRatio="none"></svg>
    <div class="legend-note">
      <span>Solid line (circles): real L2 error vs analytical solution</span>
      <span>Dashed line (diamonds): estimator total indicator (eta)</span>
      <span>Hover points for exact values</span>
    </div>
    <div class="tooltip" id="tooltip"></div>
  </div>
</div>

<script>
const DATA = {data_json};

const CASE_LABELS = {{
  spherical_screened: "Spherical Shell (Screened)",
  spherical_unscreened: "Spherical Shell (Unscreened)",
  planar_screened: "Planar Slab (Screened)",
  planar_unscreened: "Planar Slab (Unscreened, linear exact)",
  grounded_ball_unscreened: "Grounded Ball + Charge (Unscreened)",
  half_space_screened: "Half-Space Image (Screened)",
  half_space_unscreened: "Half-Space Image (Unscreened)",
  dielectric_interface_screened: "Dielectric Interface (Screened)",
  dielectric_interface_unscreened: "Dielectric Interface (Unscreened)",
}};

const ESTIMATORS = ["kelly", "recovery", "residual", "trajectory"];
const COLORS = {{
  kelly: "var(--chart-1, #60a5fa)",
  recovery: "var(--chart-2, #34d399)",
  residual: "var(--chart-3, #f59e0b)",
  trajectory: "var(--chart-4, #f472b6)",
}};

const state = {{
  caseKey: "spherical_screened",
  enabledEst: new Set(ESTIMATORS),
  showReal: true,
  showEta: true,
  metric: "l2_rel",
  xMode: "cycle",
  yScale: "log",
}};

const caseSelect = document.getElementById("caseSelect");
const availableCases = Object.keys(CASE_LABELS).filter(k => DATA.some(d => d.case === k));
for (const k of availableCases) {{
  const opt = document.createElement("option");
  opt.value = k;
  opt.textContent = CASE_LABELS[k] || k;
  if (k === state.caseKey) opt.selected = true;
  caseSelect.appendChild(opt);
}}
caseSelect.addEventListener("change", () => {{ state.caseKey = caseSelect.value; render(); }});

const estToggles = document.getElementById("estToggles");
for (const est of ESTIMATORS) {{
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "pill";
  btn.innerHTML = `<span class="swatch" style="background:${{COLORS[est]}}"></span>${{est}}`;
  btn.addEventListener("click", () => {{
    if (state.enabledEst.has(est)) {{
      state.enabledEst.delete(est);
      btn.classList.add("off");
    }} else {{
      state.enabledEst.add(est);
      btn.classList.remove("off");
    }}
    render();
  }});
  estToggles.appendChild(btn);
}}

const toggleReal = document.getElementById("toggleReal");
const toggleEta = document.getElementById("toggleEta");
toggleReal.addEventListener("click", () => {{
  state.showReal = !state.showReal;
  toggleReal.classList.toggle("off", !state.showReal);
  render();
}});
toggleEta.addEventListener("click", () => {{
  state.showEta = !state.showEta;
  toggleEta.classList.toggle("off", !state.showEta);
  render();
}});

document.getElementById("metricSelect").addEventListener("change", (e) => {{ state.metric = e.target.value; render(); }});
document.getElementById("xSelect").addEventListener("change", (e) => {{ state.xMode = e.target.value; render(); }});
document.getElementById("yScaleSelect").addEventListener("change", (e) => {{ state.yScale = e.target.value; render(); }});

const svg = document.getElementById("svg");
const tooltip = document.getElementById("tooltip");

function fmtSci(v) {{
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) return v.toPrecision(3);
  return v.toExponential(2);
}}

function render() {{
  const rows = DATA.filter(d => d.case === state.caseKey && state.enabledEst.has(d.estimator));
  document.getElementById("chartTitle").textContent = CASE_LABELS[state.caseKey] || state.caseKey;
  const maxEls = rows.reduce((m, r) => Math.max(m, r.n_elements), 0);
  document.getElementById("chartSub").textContent = maxEls ? `up to ${{maxEls.toLocaleString()}} tets` : "no series selected";

  const W = 700, H = 360;
  const pad = {{ left: 68, right: 20, top: 14, bottom: 42 }};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  const yVals = [];
  const xVals = [];
  for (const r of rows) {{
    xVals.push(state.xMode === "cycle" ? r.cycle : r.n_elements);
    if (state.showReal && Number.isFinite(r[state.metric])) {{
      const v = r[state.metric];
      if (state.yScale !== "log" || v > 0) yVals.push(v);
    }}
    if (state.showEta && r.eta_total !== null && Number.isFinite(r.eta_total)) {{
      const v = r.eta_total;
      if (state.yScale !== "log" || v > 0) yVals.push(v);
    }}
  }}

  svg.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {{}}, text = null) => {{
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== null) n.textContent = text;
    return n;
  }};

  if (!yVals.length || !xVals.length) {{
    svg.appendChild(el("text", {{ x: W / 2, y: H / 2, "text-anchor": "middle", class: "axis-title" }}, "Toggle on at least one estimator and series"));
    return;
  }}

  let xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  if (xMin === xMax) {{ xMin -= 1; xMax += 1; }}
  const xMap = (v) => {{
    if (state.xMode === "n_elements") {{
      const lx0 = Math.log10(Math.max(1, xMin)), lx1 = Math.log10(Math.max(1, xMax));
      const span = lx1 - lx0 || 1;
      return pad.left + ((Math.log10(Math.max(1, v)) - lx0) / span) * pw;
    }}
    return pad.left + ((v - xMin) / (xMax - xMin)) * pw;
  }};

  let yMin = Math.min(...yVals), yMax = Math.max(...yVals);
  let yMap, yTicks = [];
  if (state.yScale === "log") {{
    let lo = Math.floor(Math.log10(Math.max(1e-18, yMin)));
    let hi = Math.ceil(Math.log10(Math.max(1e-18, yMax)));
    if (lo === hi) {{ lo -= 1; hi += 1; }}
    yMap = (v) => pad.top + ph - ((Math.log10(Math.max(1e-18, v)) - lo) / (hi - lo)) * ph;
    const step = (hi - lo) > 8 ? 2 : 1;
    for (let p = lo; p <= hi; p += step) yTicks.push({{ v: Math.pow(10, p), label: `1e${{p}}` }});
  }} else {{
    const span = (yMax - yMin) || Math.abs(yMax) || 1;
    const lo = Math.max(0, yMin - 0.05 * span);
    const hi = yMax + 0.08 * span;
    yMap = (v) => pad.top + ph - ((v - lo) / (hi - lo)) * ph;
    for (let i = 0; i <= 5; i++) {{
      const v = lo + (i / 5) * (hi - lo);
      yTicks.push({{ v, label: fmtSci(v) }});
    }}
  }}

  // Grid + Y ticks
  for (const t of yTicks) {{
    const y = yMap(t.v);
    svg.appendChild(el("line", {{ x1: pad.left, x2: W - pad.right, y1: y, y2: y, class: "grid-line" }}));
    svg.appendChild(el("text", {{ x: pad.left - 8, y: y + 4, "text-anchor": "end", class: "axis-label" }}, t.label));
  }}

  // X ticks
  let xTicks = [];
  if (state.xMode === "cycle") {{
    const uniq = [...new Set(xVals)].sort((a, b) => a - b);
    xTicks = uniq.map(v => ({{ v, label: String(v) }}));
  }} else {{
    const lx0 = Math.log10(Math.max(1, xMin)), lx1 = Math.log10(Math.max(1, xMax));
    for (let i = 0; i <= 4; i++) {{
      const v = Math.round(Math.pow(10, lx0 + (i / 4) * (lx1 - lx0)));
      xTicks.push({{ v, label: v.toLocaleString() }});
    }}
  }}
  for (const t of xTicks) {{
    const x = xMap(t.v);
    svg.appendChild(el("line", {{ x1: x, x2: x, y1: pad.top, y2: pad.top + ph, class: "grid-line", "stroke-dasharray": "2 4" }}));
    svg.appendChild(el("text", {{ x, y: pad.top + ph + 16, "text-anchor": "middle", class: "axis-label" }}, t.label));
  }}

  // Axis titles
  svg.appendChild(el("text", {{ x: pad.left + pw / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" }},
    state.xMode === "cycle" ? "AMR Iteration (cycle)" : "Number of Tetrahedra (log scale)"));
  const yTitle = state.metric === "l2_rel" ? "Error (Relative L2 / eta)" : "Error (Absolute L2 / eta)";
  svg.appendChild(el("text", {{
    x: 14, y: pad.top + ph / 2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 14 ${{pad.top + ph / 2}})`
  }}, yTitle));

  const showTip = (evt, text) => {{
    const rect = svg.parentElement.getBoundingClientRect();
    tooltip.textContent = text;
    tooltip.style.display = "block";
    const left = Math.min(rect.width - tooltip.offsetWidth - 10, Math.max(8, evt.clientX - rect.left + 12));
    const top = Math.max(8, evt.clientY - rect.top - tooltip.offsetHeight - 8);
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }};
  const hideTip = () => {{ tooltip.style.display = "none"; }};

  // Draw series per estimator
  for (const est of ESTIMATORS) {{
    if (!state.enabledEst.has(est)) continue;
    const sRows = rows.filter(r => r.estimator === est).sort((a, b) => a.cycle - b.cycle);
    if (!sRows.length) continue;
    const color = COLORS[est];

    // Real L2 error (solid)
    if (state.showReal) {{
      const pts = sRows.filter(r => Number.isFinite(r[state.metric]) && (state.yScale !== "log" || r[state.metric] > 0));
      if (pts.length > 1) {{
        const d = pts.map((r, i) => `${{i === 0 ? "M" : "L"}}${{xMap(state.xMode === "cycle" ? r.cycle : r.n_elements).toFixed(1)}},${{yMap(r[state.metric]).toFixed(1)}}`).join(" ");
        svg.appendChild(el("path", {{ d, fill: "none", stroke: color, "stroke-width": 2.2 }}));
      }}
      for (const r of pts) {{
        const cx = xMap(state.xMode === "cycle" ? r.cycle : r.n_elements);
        const cy = yMap(r[state.metric]);
        const c = el("circle", {{ cx, cy, r: 4, fill: color, stroke: "var(--card, #111)", "stroke-width": 1.2, style: "cursor:pointer" }});
        c.addEventListener("mousemove", (e) => showTip(e,
          `${{est}} | cycle ${{r.cycle}}\\ntets: ${{r.n_elements.toLocaleString()}}, nodes: ${{r.n_nodes.toLocaleString()}}\\nrel L2: ${{fmtSci(r.l2_rel)}}\\nabs L2: ${{fmtSci(r.l2_abs)}}\\neta:    ${{fmtSci(r.eta_total)}}`));
        c.addEventListener("mouseleave", hideTip);
        svg.appendChild(c);
      }}
    }}

    // Estimated error eta (dashed)
    if (state.showEta) {{
      const pts = sRows.filter(r => r.eta_total !== null && Number.isFinite(r.eta_total) && (state.yScale !== "log" || r.eta_total > 0));
      if (pts.length > 1) {{
        const d = pts.map((r, i) => `${{i === 0 ? "M" : "L"}}${{xMap(state.xMode === "cycle" ? r.cycle : r.n_elements).toFixed(1)}},${{yMap(r.eta_total).toFixed(1)}}`).join(" ");
        svg.appendChild(el("path", {{ d, fill: "none", stroke: color, "stroke-width": 1.8, "stroke-dasharray": "5 4", opacity: 0.85 }}));
      }}
      for (const r of pts) {{
        const cx = xMap(state.xMode === "cycle" ? r.cycle : r.n_elements);
        const cy = yMap(r.eta_total);
        const d = `M${{cx}},${{cy - 4}} L${{cx + 4}},${{cy}} L${{cx}},${{cy + 4}} L${{cx - 4}},${{cy}} Z`;
        const p = el("path", {{ d, fill: color, opacity: 0.85, stroke: "var(--card, #111)", "stroke-width": 1, style: "cursor:pointer" }});
        p.addEventListener("mousemove", (e) => showTip(e,
          `${{est}} (eta) | cycle ${{r.cycle}}\\ntets: ${{r.n_elements.toLocaleString()}}, nodes: ${{r.n_nodes.toLocaleString()}}\\neta:    ${{fmtSci(r.eta_total)}}\\nrel L2: ${{fmtSci(r.l2_rel)}}`));
        p.addEventListener("mouseleave", hideTip);
        svg.appendChild(p);
      }}
    }}
  }}
}}

render();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    # 1. Load the saved experiment run (no solves or AMR stages re-run).
    saved = load_run(RESULTS_DIR)
    print(f"Loaded {len(saved.successful)}/{len(saved)} successful branches from {RESULTS_DIR}")

    per_cycle_rows: list[dict[str, Any]] = []

    # 2. Define the per-branch analysis function passed to `evaluate(...)`.
    def analyze_branch(run: RunContext, branch: BranchResult) -> dict[str, Any]:
        case: Case = branch["case"]
        case_label = branch.choice_labels["case"]
        estimator = branch.choice_labels["estimator"]
        history: list[dict[str, Any]] = branch["history"]

        cycle_metrics: list[dict[str, Any]] = []
        for entry in history:
            mesh_data = read_mesh_properties(entry["mesh"].path)
            field_data = read_nodal_field(entry["field"].path)
            err = compute_l2_error(case, mesh_data, field_data.u)
            eta_total = extract_eta_total(entry.get("adapt_info"))

            row = {
                "case": case_label,
                "estimator": estimator,
                "cycle": int(entry["cycle"]),
                "n_elements": int(len(mesh_data.elements)),
                "n_nodes": int(len(mesh_data.nodes)),
                "l2_abs": err["l2_abs"],
                "l2_rel": err["l2_rel"],
                "eta_total": eta_total,
            }
            cycle_metrics.append(row)
            if not run.dry_run:
                per_cycle_rows.append(row)

        first = cycle_metrics[0]
        last = cycle_metrics[-1]
        return {
            "cycles": len(cycle_metrics),
            "initial_elements": first["n_elements"],
            "final_elements": last["n_elements"],
            "initial_l2_rel": first["l2_rel"],
            "final_l2_rel": last["l2_rel"],
            "initial_l2_abs": first["l2_abs"],
            "final_l2_abs": last["l2_abs"],
            "error_reduction": (first["l2_rel"] / last["l2_rel"]) if last["l2_rel"] > 0 else float("inf"),
        }

    # 3. Run the post-hoc evaluation via `evaluate(...)`.
    ana_report = evaluate(saved, analyze_branch)
    ana_report.save(ANALYSIS_DIR, overwrite=True)

    # 4. Save the tidy per-cycle CSV and interactive HTML chart.
    df_cycles = pd.DataFrame(per_cycle_rows)
    df_cycles.to_csv(ANALYSIS_DIR / "convergence_by_cycle.csv", index=False)

    chart_path = ANALYSIS_DIR / "convergence_chart.html"
    build_interactive_html(per_cycle_rows, chart_path)

    # Print a concise summary table of initial vs final relative L2 error.
    df_summary = ana_report.to_dataframe()
    cols = ["case", "estimator", "res.cycles", "res.initial_elements", "res.final_elements", "res.initial_l2_rel", "res.final_l2_rel"]
    print("\nSummary (initial -> final relative L2 error):")
    print(df_summary[cols].to_string(index=False))
    print(f"\nSaved analysis report to: {ANALYSIS_DIR}")
    print(f"Saved interactive chart to: {chart_path}")


if __name__ == "__main__":
    main()
