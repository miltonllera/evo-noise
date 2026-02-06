#pragma once

#include "types.hpp"
#include "rng.hpp"

namespace evo {

class GillespieSimulator {
public:
    GillespieSimulator(const GeneExpressionParams& params, PCG64& rng);

    void compute_propensities(const GeneExpressionState& state, double* propensities) const;
    void apply_reaction(GeneExpressionState& state, int reaction_idx) const;
    double step(GeneExpressionState& state);
    int simulate_until(GeneExpressionState& state, double target_time, int max_reactions);

private:
    GeneExpressionParams params_;
    PCG64& rng_;
};

}  // namespace evo
