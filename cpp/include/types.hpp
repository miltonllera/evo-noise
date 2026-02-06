#pragma once

#include <cstdint>
#include <array>
#include <vector>

namespace evo {

enum class TileType : int8_t {
    EMPTY = 0,
    FOOD = 1,
    POISON = 2
};

struct GeneExpressionParams {
    float k_transcription;
    float k_translation;
    float k_mrna_deg;
    float k_protein_deg;
};

struct GeneExpressionState {
    int mrna;
    int protein;
    double time;
};

struct PerceptionConfig {
    int window_size;
    int hidden_size;
    float k_transcription_min;
    float k_transcription_max;
    float k_translation_min;
    float k_translation_max;
    float k_mrna_deg_min;
    float k_mrna_deg_max;
    float k_protein_deg_min;
    float k_protein_deg_max;
    float mutation_rate;
    float mutation_std;
};

struct TargetDistribution {
    float mean;
    float std;
};

struct ActionTargets {
    TargetDistribution move_up;
    TargetDistribution move_down;
    TargetDistribution move_left;
    TargetDistribution move_right;
    TargetDistribution stay;
};

struct GaussianComponent {
    float mean_x;
    float mean_y;
    float variance;
    float weight;
};

struct FoodDistributionConfig {
    std::vector<GaussianComponent> components;
    int total_food_per_step;
};

struct EnvironmentConfig {
    int width;
    int height;
    float food_spawn_prob;
    float poison_spawn_prob;
    float food_energy;
    float poison_energy;
    float move_cost;
    float reproduction_threshold;
    float reproduction_cost;
    float base_metabolism;
    float gene_expression_dt;
    bool use_gene_expression;
    float protein_low;
    float protein_high;
    PerceptionConfig perception_config;
    bool use_distribution_mapper;
    int history_window_size;
    int history_min_samples;
    ActionTargets action_targets;
    float action_temperature;
    bool use_gmm_food;
    FoodDistributionConfig food_distribution;
};

inline PerceptionConfig default_perception_config() {
    return PerceptionConfig{
        .window_size = 5,
        .hidden_size = 16,
        .k_transcription_min = 0.1f,
        .k_transcription_max = 2.0f,
        .k_translation_min = 0.1f,
        .k_translation_max = 5.0f,
        .k_mrna_deg_min = 0.01f,
        .k_mrna_deg_max = 0.5f,
        .k_protein_deg_min = 0.005f,
        .k_protein_deg_max = 0.1f,
        .mutation_rate = 0.1f,
        .mutation_std = 0.1f
    };
}

inline ActionTargets default_action_targets() {
    return ActionTargets{
        .move_up = {100.0f, 50.0f},
        .move_down = {100.0f, 50.0f},
        .move_left = {100.0f, 50.0f},
        .move_right = {100.0f, 50.0f},
        .stay = {200.0f, 20.0f}
    };
}

inline EnvironmentConfig default_environment_config() {
    return EnvironmentConfig{
        .width = 50,
        .height = 50,
        .food_spawn_prob = 0.01f,
        .poison_spawn_prob = 0.005f,
        .food_energy = 30.0f,
        .poison_energy = -50.0f,
        .move_cost = 1.0f,
        .reproduction_threshold = 150.0f,
        .reproduction_cost = 60.0f,
        .base_metabolism = 0.5f,
        .gene_expression_dt = 1.0f,
        .use_gene_expression = true,
        .protein_low = 50.0f,
        .protein_high = 200.0f,
        .perception_config = default_perception_config(),
        .use_distribution_mapper = false,
        .history_window_size = 50,
        .history_min_samples = 10,
        .action_targets = default_action_targets(),
        .action_temperature = 1.0f,
        .use_gmm_food = false,
        .food_distribution = {}
    };
}

}  // namespace evo
