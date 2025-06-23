#pragma once

namespace tauv::math_utils {

    double ipow2(double x) {
        return x * x;
    }

    template<int N>
    double ipow(double x) {
        static_assert(N >= 0, "Exponent must be non-negative");
        if constexpr (N == 0) return 1.0;
        else return x * ipow<N-1>(x);
    }

}