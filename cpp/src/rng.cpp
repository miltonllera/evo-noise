#include "rng.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace evo {

PCG64::PCG64(uint64_t seed) : PCG64(seed, 1) {}

PCG64::PCG64(uint64_t seed, uint64_t stream)
    : state_(0), inc_((stream << 1) | 1), has_spare_normal_(false), spare_normal_(0.0) {
    next();
    state_ += seed;
    next();
}

uint64_t PCG64::next() {
    uint64_t oldstate = state_;
    state_ = oldstate * MULTIPLIER + inc_;
    // PCG-XSH-RR: produces 32-bit output from 64-bit state
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

double PCG64::uniform() {
    // PCG-XSH-RR produces 32-bit output
    return static_cast<double>(next()) / 4294967296.0;  // 2^32
}

double PCG64::uniform(double low, double high) {
    return low + uniform() * (high - low);
}

double PCG64::exponential(double scale) {
    double u = uniform();
    while (u == 0.0) {
        u = uniform();
    }
    return -scale * std::log(u);
}

double PCG64::normal(double mean, double std) {
    if (has_spare_normal_) {
        has_spare_normal_ = false;
        return mean + std * spare_normal_;
    }

    double u, v, s;
    do {
        u = uniform() * 2.0 - 1.0;
        v = uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double mul = std::sqrt(-2.0 * std::log(s) / s);
    spare_normal_ = v * mul;
    has_spare_normal_ = true;
    return mean + std * u * mul;
}

int PCG64::poisson(double lambda) {
    if (lambda < 30.0) {
        double L = std::exp(-lambda);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= uniform();
        } while (p > L);
        return k - 1;
    } else {
        double c = 0.767 - 3.36 / lambda;
        double beta = M_PI / std::sqrt(3.0 * lambda);
        double alpha = beta * lambda;
        double k_val = std::log(c) - lambda - std::log(beta);

        while (true) {
            double u = uniform();
            double x = (alpha - std::log((1.0 - u) / u)) / beta;
            int n = static_cast<int>(std::floor(x + 0.5));
            if (n < 0) continue;

            double v = uniform();
            double y = alpha - beta * x;
            double temp = 1.0 + std::exp(y);
            double lhs = y + std::log(v / (temp * temp));
            double rhs = k_val + n * std::log(lambda) - std::lgamma(n + 1);
            if (lhs <= rhs) return n;
        }
    }
}

int PCG64::integers(int low, int high) {
    if (high <= low) return low;
    uint64_t range = static_cast<uint64_t>(high - low);
    uint64_t threshold = -range % range;
    uint64_t r;
    do {
        r = next();
    } while (r < threshold);
    return low + static_cast<int>(r % range);
}

int PCG64::choice(int n, const double* weights) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += weights[i];
    }

    if (total == 0.0) {
        return integers(0, n);
    }

    double r = uniform() * total;
    double cumsum = 0.0;
    for (int i = 0; i < n; i++) {
        cumsum += weights[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

int PCG64::choice(int n, const std::vector<double>& weights) {
    return choice(n, weights.data());
}

void PCG64::seed(uint64_t new_seed) {
    state_ = 0;
    next();
    state_ += new_seed;
    next();
    has_spare_normal_ = false;
}

PCG64 PCG64::spawn() const {
    uint64_t new_seed = const_cast<PCG64*>(this)->next();
    uint64_t new_stream = const_cast<PCG64*>(this)->next();
    return PCG64(new_seed, new_stream);
}

}  // namespace evo
