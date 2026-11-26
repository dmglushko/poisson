"""Physical constants matching include/common/constants.h (CODATA 2022, Nebula units)."""

from __future__ import annotations

import math

NM_TO_M: float = 1e-9
M_TO_NM: float = 1e9

ELEMENTARY_CHARGE: float = 1.602176634e-19       # e [C] (exact)
VACUUM_PERMITTIVITY: float = 8.8541878188e-12    # eps_0 [F/m] (CODATA 2022)

# RHS factor for -div(eps_r grad u) + c u = rho / eps_0 with lengths in nm and u in V.
E_OVER_EPS0_NM: float = ELEMENTARY_CHARGE / (VACUUM_PERMITTIVITY * NM_TO_M)

# Coulomb prefactor e / (4 * pi * eps_0 * 1 nm) [V * nm].
COULOMB_CONSTANT_NM: float = E_OVER_EPS0_NM / (4.0 * math.pi)
