#pragma once

#include "types.hpp"
#include "rng.hpp"
#include "perception.hpp"
#include "action_mapper.hpp"
#include "cell.hpp"
#include <vector>
#include <memory>

namespace evo {

class Environment {
public:
    Environment(const EnvironmentConfig& config, uint64_t seed);

    Cell& add_cell(int x, int y, double energy);
    void spawn_random_cells(int n, double energy);
    void step();
    void run(int n_steps);

    // Accessors
    int timestep() const { return timestep_; }
    int width() const { return config_.width; }
    int height() const { return config_.height; }
    const std::vector<TileType>& grid() const { return grid_; }
    const std::vector<Cell>& cells() const { return cells_; }
    int count_food() const;
    int count_poison() const;

    // For output
    const EnvironmentConfig& config() const { return config_; }

private:
    EnvironmentConfig config_;
    PCG64 rng_;
    std::vector<TileType> grid_;  // Row-major: grid_[y * width + x]
    std::vector<Cell> cells_;
    int timestep_;

    std::unique_ptr<PerceptionSystem> perception_system_;
    std::unique_ptr<ActionMapper> action_mapper_;
    std::unique_ptr<DistributionActionMapper> dist_action_mapper_;

    void spawn_resources();
    void spawn_food_uniform();
    void spawn_food_gmm();
    void spawn_poison();

    void move_cell(Cell& cell);
    void cell_consume(Cell& cell);
    Cell* cell_reproduce(Cell& cell, std::vector<Cell>& new_cells);
    void apply_metabolism(Cell& cell);
    void update_gene_params_from_perception(Cell& cell);
    void update_gene_expression(Cell& cell);

    TileType& grid_at(int x, int y);
    TileType grid_at(int x, int y) const;
};

}  // namespace evo
