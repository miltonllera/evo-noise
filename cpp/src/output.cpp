#include "output.hpp"
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace evo {

OutputWriter::OutputWriter(const std::string& filename)
    : file_(filename, std::ios::binary), closed_(false) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
}

OutputWriter::~OutputWriter() {
    close();
}

void OutputWriter::write_frame(const Environment& env) {
    if (closed_) return;

    const auto& cells = env.cells();
    double total_energy = 0.0;
    double total_protein = 0.0;
    for (const auto& cell : cells) {
        total_energy += cell.energy;
        total_protein += cell.get_protein();
    }
    double mean_energy = cells.empty() ? 0.0 : total_energy / cells.size();
    double mean_protein = cells.empty() ? 0.0 : total_protein / cells.size();

    FrameHeader header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.timestep = static_cast<uint32_t>(env.timestep());
    header.n_cells = static_cast<uint32_t>(cells.size());
    header.width = static_cast<uint32_t>(env.width());
    header.height = static_cast<uint32_t>(env.height());
    header.food_count = static_cast<uint32_t>(env.count_food());
    header.poison_count = static_cast<uint32_t>(env.count_poison());
    header.total_energy = total_energy;
    header.mean_energy = mean_energy;
    header.mean_protein = mean_protein;

    file_.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write grid
    const auto& grid = env.grid();
    file_.write(reinterpret_cast<const char*>(grid.data()), grid.size() * sizeof(TileType));

    // Write cells
    for (const auto& cell : cells) {
        CellData data;
        data.cell_id = cell.id;
        data.x = cell.x;
        data.y = cell.y;
        data.energy = cell.energy;
        data.protein = cell.get_protein();
        data.mrna = cell.get_mrna();
        data.age = cell.age;
        data.k_transcription = cell.gene_params.k_transcription;
        data.k_translation = cell.gene_params.k_translation;
        data.k_mrna_deg = cell.gene_params.k_mrna_deg;
        data.k_protein_deg = cell.gene_params.k_protein_deg;
        file_.write(reinterpret_cast<const char*>(&data), sizeof(data));
    }
}

void OutputWriter::close() {
    if (!closed_ && file_.is_open()) {
        file_.close();
        closed_ = true;
    }
}

FrameOutputWriter::FrameOutputWriter(const std::string& base_path)
    : base_path_(base_path) {}

void FrameOutputWriter::write_frame(const Environment& env, int frame_number) {
    std::ostringstream filename;
    filename << base_path_ << "_" << std::setfill('0') << std::setw(6) << frame_number << ".bin";

    OutputWriter writer(filename.str());
    writer.write_frame(env);
}

}  // namespace evo
