#pragma once

#include "mesh_properties.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Marking: the second of the three parts of adaptive refinement (estimate -> mark ->
// refine). A marking strategy picks the tets to refine from the error indicators.
//
// mark() is not virtual: it checks the indicators and applies the rules every strategy
// shares, then asks the strategy (select()) which tets it wants. Each strategy has its own
// Settings, given to its constructor and checked there (std::invalid_argument).
class MarkingStrategy {
public:
    virtual ~MarkingStrategy() = default;
    virtual std::string name() const = 0;

    // The marked tets of `mesh`, in increasing index order, without repetitions. `eta` holds
    // one indicator per tet (eta[k] belongs to mesh.elements[k]), each finite and
    // non-negative; otherwise std::invalid_argument, naming the tet. A tet with eta_K = 0 is
    // never marked, so all zeros mark nothing. Deterministic: equal indicators are ordered
    // by tet index.
    std::vector<uint32_t> mark(const MeshProperties& mesh, std::span<const double> eta) const;

private:
    // Called by mark() with valid indicators, at least one of them positive. Returns the
    // tets the strategy picks, in any order and possibly with zero indicators; mark() drops
    // those and sorts.
    virtual std::vector<uint32_t> select(std::span<const double> eta) const = 0;
};

// Dörfler (bulk) marking: the smallest set M of tets with
//   sum over M of eta_K^2 >= theta * sum over all K of eta_K^2,
// taken largest indicator first.
class DorflerMarking final : public MarkingStrategy {
public:
    struct Settings {
        double theta = 0.5;                       // in [0, 1]; 0 marks nothing, 1 every positive one
    };
    DorflerMarking();   // default Settings
    explicit DorflerMarking(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "dorfler"; }

private:
    std::vector<uint32_t> select(std::span<const double> eta) const override;
    Settings settings_;
};

// The maximum strategy: every tet with eta_K >= theta * max eta.
class MaximumMarking final : public MarkingStrategy {
public:
    struct Settings {
        double theta = 0.5;                       // in [0, 1]; 1 marks only the largest, 0 every positive one
    };
    MaximumMarking();   // default Settings
    explicit MaximumMarking(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "maximum"; }

private:
    std::vector<uint32_t> select(std::span<const double> eta) const override;
    Settings settings_;
};

// Fixed fraction marking: the round(fraction * N) tets with the largest indicators, N the
// number of tets (fewer if some of them have eta_K = 0).
class FixedFractionMarking final : public MarkingStrategy {
public:
    struct Settings {
        double fraction = 0.3;                    // in [0, 1]
    };
    FixedFractionMarking();   // default Settings
    explicit FixedFractionMarking(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "fixed_fraction"; }

private:
    std::vector<uint32_t> select(std::span<const double> eta) const override;
    Settings settings_;
};

// Equidistribution marking: every tet with eta_K >= tolerance / sqrt(N). If every indicator
// were exactly at that bound, the total would be the tolerance.
class EquidistributionMarking final : public MarkingStrategy {
public:
    struct Settings {
        double tolerance = 0;                     // must be set: finite and positive, in the units of eta
    };
    explicit EquidistributionMarking(Settings settings);
    const Settings& settings() const { return settings_; }
    std::string name() const override { return "equidistribution"; }

private:
    std::vector<uint32_t> select(std::span<const double> eta) const override;
    Settings settings_;
};
