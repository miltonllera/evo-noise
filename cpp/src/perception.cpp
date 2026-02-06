#include "perception.hpp"
#include <cmath>
#include <algorithm>

namespace evo {

PerceptionNetwork PerceptionNetwork::random_init(int hidden_size, PCG64& rng) {
    PerceptionNetwork net;
    net.hidden_size = hidden_size;

    // Xavier initialization
    float w1_scale = std::sqrt(2.0f / (INPUT_SIZE + hidden_size));
    float w2_scale = std::sqrt(2.0f / (hidden_size + OUTPUT_SIZE));

    net.w1.resize(INPUT_SIZE * hidden_size);
    for (int i = 0; i < INPUT_SIZE * hidden_size; i++) {
        net.w1[i] = static_cast<float>(rng.normal(0.0, w1_scale));
    }

    net.b1.resize(hidden_size, 0.0f);

    net.w2.resize(hidden_size * OUTPUT_SIZE);
    for (int i = 0; i < hidden_size * OUTPUT_SIZE; i++) {
        net.w2[i] = static_cast<float>(rng.normal(0.0, w2_scale));
    }

    net.b2.resize(OUTPUT_SIZE, 0.0f);

    return net;
}

PerceptionNetwork PerceptionNetwork::copy() const {
    PerceptionNetwork net;
    net.hidden_size = hidden_size;
    net.w1 = w1;
    net.b1 = b1;
    net.w2 = w2;
    net.b2 = b2;
    return net;
}

void PerceptionNetwork::mutate(float mutation_rate, float mutation_std, PCG64& rng) {
    auto mutate_array = [&](std::vector<float>& arr) {
        for (size_t i = 0; i < arr.size(); i++) {
            if (rng.uniform() < mutation_rate) {
                arr[i] += static_cast<float>(rng.normal(0.0, mutation_std));
            }
        }
    };

    mutate_array(w1);
    mutate_array(b1);
    mutate_array(w2);
    mutate_array(b2);
}

void extract_local_window(
    const TileType* grid,
    int grid_width,
    int grid_height,
    int x,
    int y,
    int window_size,
    TileType* window
) {
    int half = window_size / 2;
    int idx = 0;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int wx = ((x + dx) % grid_width + grid_width) % grid_width;
            int wy = ((y + dy) % grid_height + grid_height) % grid_height;
            window[idx++] = grid[wy * grid_width + wx];
        }
    }
}

void perceive_environment(
    const TileType* grid,
    int grid_width,
    int grid_height,
    int x,
    int y,
    int window_size,
    float* features
) {
    int window_area = window_size * window_size;
    std::vector<TileType> window(window_area);
    extract_local_window(grid, grid_width, grid_height, x, y, window_size, window.data());

    // Food channel
    for (int i = 0; i < window_area; i++) {
        features[i] = (window[i] == TileType::FOOD) ? 1.0f : 0.0f;
    }

    // Poison channel
    for (int i = 0; i < window_area; i++) {
        features[window_area + i] = (window[i] == TileType::POISON) ? 1.0f : 0.0f;
    }
}

float softplus(float x) {
    if (x > 20.0f) return x;
    x = std::max(-20.0f, std::min(20.0f, x));
    return std::log1p(std::exp(x));
}

static float scale_to_range(float val, float min_val, float max_val) {
    float scaled = val / (1.0f + val);
    return min_val + scaled * (max_val - min_val);
}

GeneExpressionParams forward_pass(
    const float* features,
    const PerceptionNetwork& network,
    const PerceptionConfig& config
) {
    int hidden_size = network.hidden_size;

    // Hidden layer with ReLU
    std::vector<float> hidden(hidden_size);
    for (int j = 0; j < hidden_size; j++) {
        float sum = network.b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * network.w1[i * hidden_size + j];
        }
        hidden[j] = std::max(0.0f, sum);  // ReLU
    }

    // Output layer with softplus
    float raw_output[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        float sum = network.b2[j];
        for (int i = 0; i < hidden_size; i++) {
            sum += hidden[i] * network.w2[i * OUTPUT_SIZE + j];
        }
        raw_output[j] = softplus(sum);
    }

    // Scale to parameter ranges
    return GeneExpressionParams{
        .k_transcription = scale_to_range(raw_output[0], config.k_transcription_min, config.k_transcription_max),
        .k_translation = scale_to_range(raw_output[1], config.k_translation_min, config.k_translation_max),
        .k_mrna_deg = scale_to_range(raw_output[2], config.k_mrna_deg_min, config.k_mrna_deg_max),
        .k_protein_deg = scale_to_range(raw_output[3], config.k_protein_deg_min, config.k_protein_deg_max)
    };
}

PerceptionSystem::PerceptionSystem(const PerceptionConfig& config, PCG64& rng)
    : config_(config), rng_(rng) {
    int window_area = config.window_size * config.window_size;
    features_buffer_.resize(window_area * 2);  // 2 channels
}

PerceptionNetwork PerceptionSystem::create_network() {
    return PerceptionNetwork::random_init(config_.hidden_size, rng_);
}

GeneExpressionParams PerceptionSystem::perceive(
    const TileType* grid,
    int grid_width,
    int grid_height,
    int x,
    int y,
    const PerceptionNetwork& network
) {
    perceive_environment(grid, grid_width, grid_height, x, y, config_.window_size, features_buffer_.data());
    return forward_pass(features_buffer_.data(), network, config_);
}

PerceptionNetwork PerceptionSystem::reproduce_network(const PerceptionNetwork& parent) {
    PerceptionNetwork offspring = parent.copy();
    offspring.mutate(config_.mutation_rate, config_.mutation_std, rng_);
    return offspring;
}

}  // namespace evo
