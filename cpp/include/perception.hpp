#pragma once

#include "types.hpp"
#include "rng.hpp"
#include <vector>
#include <array>

namespace evo {

constexpr int INPUT_SIZE = 50;   // 2 channels * 25 values (5x5 window)
constexpr int OUTPUT_SIZE = 4;   // k_transcription, k_translation, k_mrna_deg, k_protein_deg

struct PerceptionNetwork {
    std::vector<float> w1;  // Shape: (INPUT_SIZE, hidden_size)
    std::vector<float> b1;  // Shape: (hidden_size,)
    std::vector<float> w2;  // Shape: (hidden_size, OUTPUT_SIZE)
    std::vector<float> b2;  // Shape: (OUTPUT_SIZE,)
    int hidden_size;

    static PerceptionNetwork random_init(int hidden_size, PCG64& rng);
    PerceptionNetwork copy() const;
    void mutate(float mutation_rate, float mutation_std, PCG64& rng);
};

void extract_local_window(
    const TileType* grid,
    int grid_width,
    int grid_height,
    int x,
    int y,
    int window_size,
    TileType* window
);

void perceive_environment(
    const TileType* grid,
    int grid_width,
    int grid_height,
    int x,
    int y,
    int window_size,
    float* features
);

float softplus(float x);

GeneExpressionParams forward_pass(
    const float* features,
    const PerceptionNetwork& network,
    const PerceptionConfig& config
);

class PerceptionSystem {
public:
    PerceptionSystem(const PerceptionConfig& config, PCG64& rng);

    PerceptionNetwork create_network();

    GeneExpressionParams perceive(
        const TileType* grid,
        int grid_width,
        int grid_height,
        int x,
        int y,
        const PerceptionNetwork& network
    );

    PerceptionNetwork reproduce_network(const PerceptionNetwork& parent);

private:
    PerceptionConfig config_;
    PCG64& rng_;
    std::vector<float> features_buffer_;
};

}  // namespace evo
