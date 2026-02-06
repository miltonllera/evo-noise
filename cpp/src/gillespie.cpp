#include "gillespie.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace evo {

GillespieSimulator::GillespieSimulator(const GeneExpressionParams& params, PCG64& rng)
    : params_(params), rng_(rng) {}

void GillespieSimulator::compute_propensities(const GeneExpressionState& state, double* propensities) const {
    propensities[0] = params_.k_transcription;
    propensities[1] = params_.k_translation * state.mrna;
    propensities[2] = params_.k_mrna_deg * state.mrna;
    propensities[3] = params_.k_protein_deg * state.protein;
}

void GillespieSimulator::apply_reaction(GeneExpressionState& state, int reaction_idx) const {
    switch (reaction_idx) {
        case 0:  // Transcription
            state.mrna += 1;
            break;
        case 1:  // Translation
            state.protein += 1;
            break;
        case 2:  // mRNA degradation
            state.mrna = std::max(0, state.mrna - 1);
            break;
        case 3:  // Protein degradation
            state.protein = std::max(0, state.protein - 1);
            break;
    }
}

double GillespieSimulator::step(GeneExpressionState& state) {
    double propensities[4];
    compute_propensities(state, propensities);

    double total_propensity = propensities[0] + propensities[1] + propensities[2] + propensities[3];

    if (total_propensity == 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    double dt = rng_.exponential(1.0 / total_propensity);

    int reaction_idx = rng_.choice(4, propensities);

    apply_reaction(state, reaction_idx);
    state.time += dt;

    return dt;
}

int GillespieSimulator::simulate_until(GeneExpressionState& state, double target_time, int max_reactions) {
    int n_reactions = 0;
    while (state.time < target_time && n_reactions < max_reactions) {
        double dt = step(state);
        n_reactions++;
        if (std::isinf(dt)) {
            state.time = target_time;
            break;
        }
    }
    return n_reactions;
}

}  // namespace evo
