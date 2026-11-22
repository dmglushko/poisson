#pragma once

#include <numbers>

// Physical constants and solver-unit conversion factors shared across pipeline stages
// (AnalyticalMaker, Assembly, Ray Tracing, and tests).
//
// Solver unit system (matching Nebula):
//   - Length:       nanometers (nm)
//   - Potential:    volts (V)
//   - Permittivity: dimensionless relative permittivity eps_r (eps_0 is folded into the RHS)
//   - Charge:       elementary charges (+1 hole, -1 electron)
//   - Energy:       electron-volts (eV)
//
// SI values follow CODATA 2022 (with exact 2019 SI definitions for e and c).
namespace poisson::constants {

// ---- Unit conversions
inline constexpr double nm_to_m = 1e-9;   // m / nm
inline constexpr double m_to_nm = 1e9;    // nm / m

// ---- SI base physical constants
inline constexpr double elementary_charge   = 1.602176634e-19;     // e [C] (exact)
inline constexpr double vacuum_permittivity = 8.8541878188e-12;    // eps_0 [F/m] (CODATA 2022)
inline constexpr double electron_mass       = 9.1093837139e-31;    // m_e [kg] (CODATA 2022)
inline constexpr double speed_of_light      = 299792458.0;         // c [m/s] (exact)

// ---- Derived solver-unit constants (nm, V, elementary charges)

// RHS factor for -div(eps_r grad u) + c u = rho / eps_0 with lengths in nm and u in V.
// A point charge of q elementary charges contributes q * e_over_eps0_nm [V * nm] to the
// integrated right-hand side.
inline constexpr double e_over_eps0_nm = elementary_charge / (vacuum_permittivity * nm_to_m);

// Coulomb prefactor e / (4 * pi * eps_0 * 1 nm) [V * nm].
// The potential of a point charge q in a uniform medium of relative permittivity eps_r at
// distance r [nm] is u(r) = (q * coulomb_constant_nm) / (eps_r * r) [V].
inline constexpr double coulomb_constant_nm = e_over_eps0_nm / (4.0 * std::numbers::pi);

// Specific charge magnitude e / m_e [C/kg = m^2 / (V * s^2)].
inline constexpr double e_over_me = elementary_charge / electron_mass;

}   // namespace poisson::constants
