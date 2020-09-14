/**
 * @file tuple.h
 * @brief about tuple
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace shirakami {

class Tuple {  // NOLINT
public:
    class Impl;

    Tuple();

    Tuple(std::string_view key, std::string_view val);

    Tuple(const Tuple &right);

    Tuple(Tuple &&right);

    Tuple &operator=(const Tuple &right);  // NOLINT
    Tuple &operator=(Tuple &&right);       // NOLINT

    [[nodiscard]] std::string_view get_key() const;    // NOLINT
    [[nodiscard]] std::string_view get_value() const;  // NOLINT
    Impl* get_pimpl();                                 // NOLINT

private:
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace shirakami
