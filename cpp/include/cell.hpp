#pragma once

#include "types.hpp"
#include "perception.hpp"
#include "action_mapper.hpp"

namespace evo {

struct Cell {
    CellId id;
    int x;
    int y;
    GeneExpressionParams gene_params;
    PerceptionNetwork perception_network;
    double energy;
    int age;
    GeneExpressionState gene_state;
    ProteinHistory protein_history;

    Cell(
        CellId id_,
        int x_,
        int y_,
        const GeneExpressionParams& gene_params_,
        PerceptionNetwork perception_network_,
        double energy_,
        int history_maxlen
    );

    void move(int dx, int dy, int grid_width, int grid_height);
    int get_protein() const;
    int get_mrna() const;
};

}  // namespace evo
