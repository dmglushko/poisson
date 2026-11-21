#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// Zero-dependency strict JSON parser for CLI --config files.
namespace poisson::cli {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    bool as_bool(const std::string& ctx) const {
        if (!is_bool()) throw std::runtime_error(ctx + ": expected boolean");
        return std::get<bool>(data);
    }
    double as_double(const std::string& ctx) const {
        if (!is_number()) throw std::runtime_error(ctx + ": expected number");
        return std::get<double>(data);
    }
    int as_int(const std::string& ctx) const {
        const double v = as_double(ctx);
        if (!std::isfinite(v) || std::floor(v) != v) throw std::runtime_error(ctx + ": expected integer");
        return static_cast<int>(v);
    }
    uint64_t as_u64(const std::string& ctx) const {
        const double v = as_double(ctx);
        if (!std::isfinite(v) || v < 0 || std::floor(v) != v)
            throw std::runtime_error(ctx + ": expected non-negative integer");
        return static_cast<uint64_t>(v);
    }
    const std::string& as_string(const std::string& ctx) const {
        if (!is_string()) throw std::runtime_error(ctx + ": expected string");
        return std::get<std::string>(data);
    }
    const Object& as_object(const std::string& ctx) const {
        if (!is_object()) throw std::runtime_error(ctx + ": expected object");
        return std::get<Object>(data);
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != text_.size()) fail("unexpected trailing characters");
        return v;
    }

private:
    std::string_view text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + msg);
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    char peek() const {
        if (pos_ >= text_.size()) fail("unexpected end of JSON");
        return text_[pos_];
    }

    JsonValue parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue{parse_string()};
        if (c == 't') {
            expect("true");
            return JsonValue{true};
        }
        if (c == 'f') {
            expect("false");
            return JsonValue{false};
        }
        if (c == 'n') {
            expect("null");
            return JsonValue{nullptr};
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
    }

    void expect(std::string_view lit) {
        if (text_.substr(pos_, lit.size()) != lit) fail("expected '" + std::string(lit) + "'");
        pos_ += lit.size();
    }

    std::string parse_string() {
        if (peek() != '"') fail("expected '\"'");
        ++pos_;
        std::string out;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= text_.size()) fail("unterminated escape sequence");
                const char e = text_[pos_++];
                switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: fail("unsupported escape sequence");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        const std::string token(text_.substr(start, pos_ - start));
        char* end = nullptr;
        const double val = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size() || !std::isfinite(val)) fail("invalid number '" + token + "'");
        return JsonValue{val};
    }

    JsonValue parse_array() {
        ++pos_;   // '['
        JsonValue::Array arr;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return JsonValue{std::move(arr)};
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            const char c = peek();
            if (c == ']') {
                ++pos_;
                return JsonValue{std::move(arr)};
            }
            if (c != ',') fail("expected ',' or ']'");
            ++pos_;
        }
    }

    JsonValue parse_object() {
        ++pos_;   // '{'
        JsonValue::Object obj;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return JsonValue{std::move(obj)};
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected ':'");
            ++pos_;
            JsonValue val = parse_value();
            obj.insert_or_assign(std::move(key), std::move(val));
            skip_ws();
            const char c = peek();
            if (c == '}') {
                ++pos_;
                return JsonValue{std::move(obj)};
            }
            if (c != ',') fail("expected ',' or '}'");
            ++pos_;
        }
    }
};

inline JsonValue parse_json_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path.string() + ": cannot open config file");
    const std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return JsonParser(content).parse();
}

}   // namespace poisson::cli
