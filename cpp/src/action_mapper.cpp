#include "action_mapper.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace evo {

ProteinHistory::ProteinHistory(int maxlen_) : maxlen(maxlen_) {}

void ProteinHistory::record(int protein) {
    buffer.push_back(protein);
    if (static_cast<int>(buffer.size()) > maxlen) {
        buffer.pop_front();
    }
}

bool ProteinHistory::is_ready(int min_samples) const {
    return static_cast<int>(buffer.size()) >= min_samples;
}

std::pair<float, float> ProteinHistory::get_stats() const {
    if (buffer.empty()) {
        return {0.0f, 0.0f};
    }

    double sum = 0.0;
    for (int val : buffer) {
        sum += val;
    }
    double mean = sum / buffer.size();

    double sq_sum = 0.0;
    for (int val : buffer) {
        double diff = val - mean;
        sq_sum += diff * diff;
    }
    double std = std::sqrt(sq_sum / buffer.size());

    return {static_cast<float>(mean), static_cast<float>(std)};
}

float normalize_protein(int protein, float low, float high) {
    if (high <= low) return 0.5f;
    float normalized = (protein - low) / (high - low);
    return std::max(0.0f, std::min(1.0f, normalized));
}

float compute_distance(float observed_mean, float observed_std, const TargetDistribution& target) {
    float mean_diff = (target.mean > 0) ? (observed_mean - target.mean) / target.mean : observed_mean;
    float std_diff = (target.std > 0) ? (observed_std - target.std) / target.std : observed_std;
    return std::sqrt(mean_diff * mean_diff + std_diff * std_diff);
}

ActionMapper::ActionMapper(float protein_low, float protein_high, PCG64& rng)
    : protein_low_(protein_low), protein_high_(protein_high), rng_(rng) {}

std::pair<int, int> ActionMapper::get_movement(int protein, float bias_strength) {
    float p_norm = normalize_protein(protein, protein_low_, protein_high_);
    float p_stay = p_norm * bias_strength;

    if (rng_.uniform() < p_stay) {
        return {0, 0};
    }

    int dx = rng_.integers(-1, 2);
    int dy = rng_.integers(-1, 2);
    return {dx, dy};
}

bool ActionMapper::should_reproduce(float energy, int protein, float base_threshold, float max_reduction) {
    float p_norm = normalize_protein(protein, protein_low_, protein_high_);
    float modifier = 1.0f - (p_norm * max_reduction);
    return energy >= base_threshold * modifier;
}

DistributionActionMapper::DistributionActionMapper(
    const ActionTargets& action_targets,
    int min_samples,
    float temperature,
    PCG64& rng
) : action_targets_(action_targets),
    min_samples_(min_samples),
    temperature_(temperature),
    rng_(rng) {}

std::pair<int, int> DistributionActionMapper::get_movement(const ProteinHistory& history) {
    if (!history.is_ready(min_samples_)) {
        int dx = rng_.integers(-1, 2);
        int dy = rng_.integers(-1, 2);
        return {dx, dy};
    }

    auto [mean, std] = history.get_stats();
    return select_action_by_sampling(mean, std);
}

bool DistributionActionMapper::should_reproduce(
    const ProteinHistory& history,
    float energy,
    float energy_threshold
) {
    (void)history;  // Unused, kept for API compatibility
    return energy >= energy_threshold;
}

std::pair<int, int> DistributionActionMapper::select_action_by_sampling(float observed_mean, float observed_std) {
    // Compute distances to each action target
    double distances[5] = {
        compute_distance(observed_mean, observed_std, action_targets_.move_up),
        compute_distance(observed_mean, observed_std, action_targets_.move_down),
        compute_distance(observed_mean, observed_std, action_targets_.move_left),
        compute_distance(observed_mean, observed_std, action_targets_.move_right),
        compute_distance(observed_mean, observed_std, action_targets_.stay)
    };

    // Softmax on negative distances (closer = higher probability)
    double logits[5];
    double max_logit = -distances[0] / temperature_;
    for (int i = 0; i < 5; i++) {
        logits[i] = -distances[i] / temperature_;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    double probs[5];
    double sum = 0.0;
    for (int i = 0; i < 5; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < 5; i++) {
        probs[i] /= sum;
    }

    int action_idx = rng_.choice(5, probs);
    return movement_direction(action_idx);
}

}  // namespace evo
