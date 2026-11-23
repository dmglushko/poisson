#pragma once

// Builds verification cases from their JSON description, {"name": ..., "params": {...}}.
// This is the only place that names the parameters; unknown names, unknown parameters,
// missing required parameters and wrong types are errors. The parameters and their defaults
// are listed in src/verification/architecture.md.

#include "cli/common/simple_json.h"
#include "verification/image_cases.h"
#include "verification/layered_cases.h"

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace poisson::cli {

using Object = JsonValue::Object;

inline void reject_unknown(const Object& obj, const std::set<std::string>& allowed, const std::string& where) {
    for (const auto& [key, value] : obj)
        if (!allowed.contains(key)) throw std::runtime_error(where + ": unknown parameter '" + key + "'");
}

inline const JsonValue* find(const Object& obj, const std::string& key) {
    const auto it = obj.find(key);
    return it == obj.end() || it->second.is_null() ? nullptr : &it->second;
}

inline const JsonValue& require(const Object& obj, const std::string& key, const std::string& where) {
    const JsonValue* v = find(obj, key);
    if (!v) throw std::runtime_error(where + ": missing parameter '" + key + "'");
    return *v;
}

inline double number_or(const Object& obj, const std::string& key, double fallback, const std::string& where) {
    const JsonValue* v = find(obj, key);
    return v ? v->as_double(where + "." + key) : fallback;
}

inline bool bool_or(const Object& obj, const std::string& key, bool fallback, const std::string& where) {
    const JsonValue* v = find(obj, key);
    return v ? v->as_bool(where + "." + key) : fallback;
}

inline std::array<double, 3> vec3(const JsonValue& v, const std::string& where) {
    if (!v.is_array()) throw std::runtime_error(where + ": expected [x, y, z]");
    const auto& a = std::get<JsonValue::Array>(v.data);
    if (a.size() != 3) throw std::runtime_error(where + ": expected [x, y, z]");
    return {a[0].as_double(where + "[0]"), a[1].as_double(where + "[1]"), a[2].as_double(where + "[2]")};
}

// {"type": "dirichlet" | "neumann" | "robin", "value": number, "alpha": number (robin only)}
inline MeshProperties::BoundaryCondition boundary_condition(const JsonValue& v, const std::string& where) {
    const Object& o = v.as_object(where);
    reject_unknown(o, {"type", "value", "alpha"}, where);
    const std::string type = require(o, "type", where).as_string(where + ".type");
    const double value = number_or(o, "value", 0, where);
    if (type == "dirichlet" || type == "neumann") {
        if (find(o, "alpha")) throw std::runtime_error(where + ": only a robin condition has alpha");
        return {type == "dirichlet" ? MeshProperties::BcType::Dirichlet : MeshProperties::BcType::Neumann, value, std::nullopt};
    }
    if (type == "robin") return {MeshProperties::BcType::Robin, value, require(o, "alpha", where).as_double(where + ".alpha")};
    throw std::runtime_error(where + ".type: unknown boundary condition '" + type + "' (dirichlet, neumann, robin)");
}

// {"name", "thickness", "eps_r", "screening_length", "metal", "voltage"}
inline std::vector<VerificationLayer> layers(const JsonValue& v, const std::string& where) {
    if (!v.is_array()) throw std::runtime_error(where + ": expected a list of layers");
    std::vector<VerificationLayer> out;
    const auto& a = std::get<JsonValue::Array>(v.data);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::string at = where + "[" + std::to_string(i) + "]";
        const Object& o = a[i].as_object(at);
        reject_unknown(o, {"name", "thickness", "eps_r", "screening_length", "metal", "voltage"}, at);
        VerificationLayer layer;
        layer.name = require(o, "name", at).as_string(at + ".name");
        layer.thickness = require(o, "thickness", at).as_double(at + ".thickness");
        layer.eps_r = number_or(o, "eps_r", 1, at);
        layer.screening_length = number_or(o, "screening_length", 0, at);
        layer.metal = bool_or(o, "metal", false, at);
        if (const JsonValue* voltage = find(o, "voltage")) layer.voltage = voltage->as_double(at + ".voltage");
        out.push_back(layer);
    }
    return out;
}

inline int32_t charge(const Object& o, const std::string& key, int32_t fallback, const std::string& where) {
    const JsonValue* v = find(o, key);
    return v ? v->as_int(where + "." + key) : fallback;
}

// Settings errors from the case constructors are std::invalid_argument; they are reported
// the same way as configuration errors.
inline std::unique_ptr<VerificationCase> make_case(const JsonValue& spec) {
    const std::string where = "case";
    const Object& root = spec.as_object(where);
    reject_unknown(root, {"name", "params"}, where);
    const std::string name = require(root, "name", where).as_string("case.name");
    static const Object no_params;
    const Object& p = find(root, "params") ? root.at("params").as_object("case.params") : no_params;
    const std::string at = "case.params";

    if (name == "planar_layers") {
        reject_unknown(p, {"layers", "left", "right", "width", "height", "screened"}, at);
        PlanarLayersCase::Settings s;
        s.layers = layers(require(p, "layers", at), at + ".layers");
        if (const JsonValue* v = find(p, "left")) s.left = boundary_condition(*v, at + ".left");
        if (const JsonValue* v = find(p, "right")) s.right = boundary_condition(*v, at + ".right");
        s.width = number_or(p, "width", s.width, at);
        s.height = number_or(p, "height", s.height, at);
        s.screened = bool_or(p, "screened", s.screened, at);
        return std::make_unique<PlanarLayersCase>(std::move(s));
    }
    if (name == "spherical_layers") {
        reject_unknown(p, {"layers", "inner_radius", "inner", "outer", "charge", "screened"}, at);
        SphericalLayersCase::Settings s;
        s.layers = layers(require(p, "layers", at), at + ".layers");
        s.inner_radius = number_or(p, "inner_radius", s.inner_radius, at);
        if (const JsonValue* v = find(p, "inner")) s.inner = boundary_condition(*v, at + ".inner");
        if (const JsonValue* v = find(p, "outer")) s.outer = boundary_condition(*v, at + ".outer");
        s.charge = charge(p, "charge", s.charge, at);
        s.screened = bool_or(p, "screened", s.screened, at);
        return std::make_unique<SphericalLayersCase>(std::move(s));
    }
    if (name == "grounded_ball_image") {
        reject_unknown(p, {"radius", "eps_r", "charge", "position"}, at);
        GroundedBallImageCase::Settings s;
        s.radius = number_or(p, "radius", s.radius, at);
        s.eps_r = number_or(p, "eps_r", s.eps_r, at);
        s.charge = charge(p, "charge", s.charge, at);
        if (const JsonValue* v = find(p, "position")) s.position = vec3(*v, at + ".position");
        return std::make_unique<GroundedBallImageCase>(s);
    }
    if (name == "half_space_image") {
        reject_unknown(p, {"eps_r", "screening_length", "charge", "distance", "half_width", "height", "plane", "screened"}, at);
        HalfSpaceImageCase::Settings s;
        s.eps_r = number_or(p, "eps_r", s.eps_r, at);
        s.screening_length = number_or(p, "screening_length", s.screening_length, at);
        s.charge = charge(p, "charge", s.charge, at);
        s.distance = number_or(p, "distance", s.distance, at);
        s.half_width = number_or(p, "half_width", s.half_width, at);
        s.height = number_or(p, "height", s.height, at);
        if (const JsonValue* v = find(p, "plane")) {
            const std::string plane = v->as_string(at + ".plane");
            if (plane == "dirichlet") s.plane = HalfSpaceImageCase::Plane::Dirichlet;
            else if (plane == "neumann") s.plane = HalfSpaceImageCase::Plane::Neumann;
            else throw std::runtime_error(at + ".plane: expected 'dirichlet' or 'neumann', not '" + plane + "'");
        }
        s.screened = bool_or(p, "screened", s.screened, at);
        return std::make_unique<HalfSpaceImageCase>(s);
    }
    if (name == "dielectric_interface") {
        reject_unknown(p, {"eps_upper", "eps_lower", "screening_length", "charge", "distance", "half_width", "half_height", "screened"}, at);
        DielectricInterfaceCase::Settings s;
        s.eps_upper = number_or(p, "eps_upper", s.eps_upper, at);
        s.eps_lower = number_or(p, "eps_lower", s.eps_lower, at);
        s.screening_length = number_or(p, "screening_length", s.screening_length, at);
        s.charge = charge(p, "charge", s.charge, at);
        s.distance = number_or(p, "distance", s.distance, at);
        s.half_width = number_or(p, "half_width", s.half_width, at);
        s.half_height = number_or(p, "half_height", s.half_height, at);
        s.screened = bool_or(p, "screened", s.screened, at);
        return std::make_unique<DielectricInterfaceCase>(s);
    }
    throw std::runtime_error("case.name: unknown case '" + name +
                             "' (planar_layers, spherical_layers, grounded_ball_image, half_space_image, dielectric_interface)");
}

}   // namespace poisson::cli
