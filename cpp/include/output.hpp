#pragma once

#include "environment.hpp"
#include <string>
#include <fstream>
#include <cstdint>

namespace evo {

// Binary output format constants
constexpr uint32_t MAGIC = 0x4E4F5645;  // "EVON" in little-endian
constexpr uint32_t VERSION = 1;

struct FrameHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t timestep;
    uint32_t n_cells;
    uint32_t width;
    uint32_t height;
    uint32_t food_count;
    uint32_t poison_count;
};

struct CellData {
    int32_t x;
    int32_t y;
    double energy;
    int32_t protein;
    int32_t mrna;
    int32_t age;
};

class OutputWriter {
public:
    explicit OutputWriter(const std::string& filename);
    ~OutputWriter();

    void write_frame(const Environment& env);
    void close();

private:
    std::ofstream file_;
    bool closed_;
};

class FrameOutputWriter {
public:
    explicit FrameOutputWriter(const std::string& base_path);

    void write_frame(const Environment& env, int frame_number);

private:
    std::string base_path_;
};

}  // namespace evo
