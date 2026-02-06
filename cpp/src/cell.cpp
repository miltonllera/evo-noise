#include "cell.hpp"

namespace evo {

Cell::Cell(
    int x_,
    int y_,
    const GeneExpressionParams& gene_params_,
    PerceptionNetwork perception_network_,
    double energy_,
    int history_maxlen
) : x(x_),
    y(y_),
    gene_params(gene_params_),
    perception_network(std::move(perception_network_)),
    energy(energy_),
    age(0),
    gene_state{0, 0, 0.0},
    protein_history(history_maxlen) {}

void Cell::move(int dx, int dy, int grid_width, int grid_height) {
    x = ((x + dx) % grid_width + grid_width) % grid_width;
    y = ((y + dy) % grid_height + grid_height) % grid_height;
}

int Cell::get_protein() const {
    return gene_state.protein;
}

int Cell::get_mrna() const {
    return gene_state.mrna;
}

}  // namespace evo
