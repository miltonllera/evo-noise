#include "environment.hpp"
#include "gillespie.hpp"
#include <algorithm>
#include <cmath>

namespace evo {

Environment::Environment(const EnvironmentConfig& config, uint64_t seed)
    : config_(config),
      rng_(seed),
      grid_(config.width * config.height, TileType::EMPTY),
      timestep_(0),
      next_cell_id_(1) {
    perception_system_ = std::make_unique<PerceptionSystem>(config_.perception_config, rng_);

    if (config_.use_distribution_mapper) {
        dist_action_mapper_ = std::make_unique<DistributionActionMapper>(
            config_.action_targets,
            config_.history_min_samples,
            config_.action_temperature,
            rng_
        );
    } else {
        action_mapper_ = std::make_unique<ActionMapper>(
            config_.protein_low,
            config_.protein_high,
            rng_
        );
    }
}

TileType& Environment::grid_at(int x, int y) {
    return grid_[y * config_.width + x];
}

TileType Environment::grid_at(int x, int y) const {
    return grid_[y * config_.width + x];
}

CellId Environment::generate_cell_id() {
    return next_cell_id_++;
}

Cell& Environment::add_cell(int x, int y, double energy) {
    PerceptionNetwork network = perception_system_->create_network();
    GeneExpressionParams params = perception_system_->perceive(
        grid_.data(), config_.width, config_.height, x, y, network
    );

    CellId id = generate_cell_id();
    cells_.emplace_back(id, x, y, params, std::move(network), energy, config_.history_window_size);
    return cells_.back();
}

void Environment::spawn_random_cells(int n, double energy) {
    for (int i = 0; i < n; i++) {
        int x = rng_.integers(0, config_.width);
        int y = rng_.integers(0, config_.height);
        add_cell(x, y, energy);
    }
}

void Environment::spawn_resources() {
    if (config_.use_gmm_food) {
        spawn_food_gmm();
    } else {
        spawn_food_uniform();
    }
    spawn_poison();
}

void Environment::spawn_food_uniform() {
    for (int y = 0; y < config_.height; y++) {
        for (int x = 0; x < config_.width; x++) {
            if (grid_at(x, y) == TileType::EMPTY) {
                if (rng_.uniform() < config_.food_spawn_prob) {
                    grid_at(x, y) = TileType::FOOD;
                }
            }
        }
    }
}

void Environment::spawn_food_gmm() {
    const auto& fd = config_.food_distribution;
    if (fd.components.empty()) return;

    // Compute total weight
    double total_weight = 0.0;
    for (const auto& comp : fd.components) {
        total_weight += comp.weight;
    }

    // Compute probability density at each grid point
    std::vector<double> density(config_.width * config_.height, 0.0);
    for (int y = 0; y < config_.height; y++) {
        for (int x = 0; x < config_.width; x++) {
            double d = 0.0;
            for (const auto& comp : fd.components) {
                double dx = x - comp.mean_x;
                double dy = y - comp.mean_y;
                double exponent = -(dx * dx + dy * dy) / (2.0 * comp.variance);
                d += (comp.weight / total_weight) * std::exp(exponent);
            }
            density[y * config_.width + x] = d;
        }
    }

    // Normalize and zero out non-empty tiles
    double sum = 0.0;
    for (int i = 0; i < config_.width * config_.height; i++) {
        if (grid_[i] != TileType::EMPTY) {
            density[i] = 0.0;
        }
        sum += density[i];
    }

    if (sum > 0.0) {
        for (double& d : density) {
            d /= sum;
        }

        int n_food = rng_.poisson(fd.total_food_per_step);
        for (int i = 0; i < n_food; i++) {
            int idx = rng_.choice(static_cast<int>(density.size()), density);
            if (grid_[idx] == TileType::EMPTY) {
                grid_[idx] = TileType::FOOD;
                density[idx] = 0.0;  // Don't spawn twice at same location
            }
        }
    }
}

void Environment::spawn_poison() {
    for (int y = 0; y < config_.height; y++) {
        for (int x = 0; x < config_.width; x++) {
            if (grid_at(x, y) == TileType::EMPTY) {
                if (rng_.uniform() < config_.poison_spawn_prob) {
                    grid_at(x, y) = TileType::POISON;
                }
            }
        }
    }
}

void Environment::update_gene_params_from_perception(Cell& cell) {
    cell.gene_params = perception_system_->perceive(
        grid_.data(), config_.width, config_.height, cell.x, cell.y, cell.perception_network
    );
}

void Environment::update_gene_expression(Cell& cell) {
    if (!config_.use_gene_expression) return;

    GillespieSimulator simulator(cell.gene_params, rng_);
    double target_time = cell.gene_state.time + config_.gene_expression_dt;
    simulator.simulate_until(cell.gene_state, target_time, 10000);
}

void Environment::move_cell(Cell& cell) {
    std::pair<int, int> movement;

    if (config_.use_gene_expression) {
        if (config_.use_distribution_mapper) {
            movement = dist_action_mapper_->get_movement(cell.protein_history);
        } else {
            movement = action_mapper_->get_movement(cell.get_protein(), 0.5f);
        }
    } else {
        movement = {rng_.integers(-1, 2), rng_.integers(-1, 2)};
    }

    cell.move(movement.first, movement.second, config_.width, config_.height);
    cell.energy -= config_.move_cost;
}

void Environment::cell_consume(Cell& cell) {
    TileType tile = grid_at(cell.x, cell.y);
    if (tile == TileType::FOOD) {
        cell.energy += config_.food_energy;
        grid_at(cell.x, cell.y) = TileType::EMPTY;
    } else if (tile == TileType::POISON) {
        cell.energy += config_.poison_energy;
        grid_at(cell.x, cell.y) = TileType::EMPTY;
    }
}

Cell* Environment::cell_reproduce(Cell& cell, std::vector<Cell>& new_cells) {
    bool should_reproduce;

    if (config_.use_gene_expression) {
        if (config_.use_distribution_mapper) {
            should_reproduce = dist_action_mapper_->should_reproduce(
                cell.protein_history, cell.energy, config_.reproduction_threshold
            );
        } else {
            should_reproduce = action_mapper_->should_reproduce(
                cell.energy, cell.get_protein(), config_.reproduction_threshold, 0.2f
            );
        }
    } else {
        should_reproduce = cell.energy >= config_.reproduction_threshold;
    }

    if (!should_reproduce) return nullptr;

    cell.energy -= config_.reproduction_cost;
    double offspring_energy = config_.reproduction_cost / 2.0;

    int dx = rng_.integers(-1, 2);
    int dy = rng_.integers(-1, 2);
    int ox = ((cell.x + dx) % config_.width + config_.width) % config_.width;
    int oy = ((cell.y + dy) % config_.height + config_.height) % config_.height;

    PerceptionNetwork offspring_network = perception_system_->reproduce_network(cell.perception_network);
    GeneExpressionParams offspring_params = perception_system_->perceive(
        grid_.data(), config_.width, config_.height, ox, oy, offspring_network
    );

    CellId offspring_id = generate_cell_id();
    new_cells.emplace_back(offspring_id, ox, oy, offspring_params, std::move(offspring_network),
                          offspring_energy, config_.history_window_size);
    return &new_cells.back();
}

void Environment::apply_metabolism(Cell& cell) {
    cell.energy -= config_.base_metabolism;
    cell.age += 1;
}

void Environment::step() {
    spawn_resources();

    std::vector<Cell> new_cells;

    for (auto& cell : cells_) {
        update_gene_params_from_perception(cell);
        update_gene_expression(cell);
        cell.protein_history.record(cell.get_protein());
        move_cell(cell);
        cell_consume(cell);
        cell_reproduce(cell, new_cells);
        apply_metabolism(cell);
    }

    cells_.insert(cells_.end(),
                  std::make_move_iterator(new_cells.begin()),
                  std::make_move_iterator(new_cells.end()));

    // Remove dead cells
    cells_.erase(
        std::remove_if(cells_.begin(), cells_.end(),
                      [](const Cell& c) { return c.energy <= 0; }),
        cells_.end()
    );

    timestep_++;
}

void Environment::run(int n_steps) {
    for (int i = 0; i < n_steps; i++) {
        step();
    }
}

int Environment::count_food() const {
    return static_cast<int>(std::count(grid_.begin(), grid_.end(), TileType::FOOD));
}

int Environment::count_poison() const {
    return static_cast<int>(std::count(grid_.begin(), grid_.end(), TileType::POISON));
}

}  // namespace evo
