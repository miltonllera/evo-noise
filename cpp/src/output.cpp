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

    FrameHeader header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.timestep = static_cast<uint32_t>(env.timestep());
    header.n_cells = static_cast<uint32_t>(env.cells().size());
    header.width = static_cast<uint32_t>(env.width());
    header.height = static_cast<uint32_t>(env.height());
    header.food_count = static_cast<uint32_t>(env.count_food());
    header.poison_count = static_cast<uint32_t>(env.count_poison());

    file_.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write grid
    const auto& grid = env.grid();
    file_.write(reinterpret_cast<const char*>(grid.data()), grid.size() * sizeof(TileType));

    // Write cells
    for (const auto& cell : env.cells()) {
        CellData data;
        data.x = cell.x;
        data.y = cell.y;
        data.energy = cell.energy;
        data.protein = cell.get_protein();
        data.mrna = cell.get_mrna();
        data.age = cell.age;
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
