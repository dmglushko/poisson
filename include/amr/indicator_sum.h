#pragma once

// How the error indicators eta_K of an estimator add up to its global estimate. An
// estimator declares it (ErrorEstimator::indicator_sum), its ErrorEstimate carries it, and
// the marking strategies that compare sums (Dörfler, equidistribution) follow it.
enum class IndicatorSum {
    // total^2 = sum of eta_K^2: estimators of a norm of the error (residual, Kelly, recovery).
    squares,
    // total = sum of eta_K: contributions that add linearly, such as the trajectory
    // estimator's contributions to an electron's landing-position error.
    values,
};
