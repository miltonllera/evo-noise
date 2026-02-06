#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

namespace evo {

class PCG64 {
public:
    explicit PCG64(uint64_t seed);
    PCG64(uint64_t seed, uint64_t stream);

    uint64_t next();
    double uniform();
    double uniform(double low, double high);
    double exponential(double scale);
    double normal(double mean, double std);
    int poisson(double lambda);
    int integers(int low, int high);
    int choice(int n, const double* weights);
    int choice(int n, const std::vector<double>& weights);

    void seed(uint64_t new_seed);
    PCG64 spawn() const;

private:
    uint64_t state_;
    uint64_t inc_;
    bool has_spare_normal_;
    double spare_normal_;

    static constexpr uint64_t MULTIPLIER = 6364136223846793005ULL;
};

}  // namespace evo
