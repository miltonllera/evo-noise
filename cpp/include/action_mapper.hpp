#pragma once

#include "types.hpp"
#include "rng.hpp"
#include <vector>
#include <deque>
#include <utility>
#include <cmath>

namespace evo {

struct ProteinHistory {
    std::deque<int> buffer;
    int maxlen;

    explicit ProteinHistory(int maxlen_);
    void record(int protein);
    bool is_ready(int min_samples) const;
    std::pair<float, float> get_stats() const;
};

float normalize_protein(int protein, float low, float high);
float compute_distance(float observed_mean, float observed_std, const TargetDistribution& target);

class ActionMapper {
public:
    ActionMapper(float protein_low, float protein_high, PCG64& rng);

    std::pair<int, int> get_movement(int protein, float bias_strength);
    bool should_reproduce(float energy, int protein, float base_threshold, float max_reduction);

private:
    float protein_low_;
    float protein_high_;
    PCG64& rng_;
};

class DistributionActionMapper {
public:
    DistributionActionMapper(
        const ActionTargets& action_targets,
        int min_samples,
        float temperature,
        PCG64& rng
    );

    std::pair<int, int> get_movement(const ProteinHistory& history);
    bool should_reproduce(const ProteinHistory& history, float energy, float energy_threshold);

private:
    ActionTargets action_targets_;
    int min_samples_;
    float temperature_;
    PCG64& rng_;

    std::pair<int, int> select_action_by_sampling(float observed_mean, float observed_std);
};

// Movement direction mapping
inline std::pair<int, int> movement_direction(int action_idx) {
    // 0: up, 1: down, 2: left, 3: right, 4: stay
    static const std::pair<int, int> directions[] = {
        {0, -1},   // up
        {0, 1},    // down
        {-1, 0},   // left
        {1, 0},    // right
        {0, 0}     // stay
    };
    return directions[action_idx];
}

}  // namespace evo
