#include "types.hpp"
#include "environment.hpp"
#include "output.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --steps N         Number of simulation steps (default: 1000)\n"
              << "  --cells N         Initial number of cells (default: 100)\n"
              << "  --width N         Grid width (default: 50)\n"
              << "  --height N        Grid height (default: 50)\n"
              << "  --seed N          Random seed (default: 42)\n"
              << "  --output FILE     Output file path (default: simulation.bin)\n"
              << "  --frame-output    Output individual frame files instead of single file\n"
              << "  --no-output       Skip writing output file\n"
              << "  --repro-threshold N  Reproduction threshold (default: 150)\n"
              << "  --quiet           Suppress progress output\n"
              << "  --help            Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    int steps = 1000;
    int initial_cells = 100;
    int width = 50;
    int height = 50;
    uint64_t seed = 42;
    std::string output_file = "simulation.bin";
    bool frame_output = false;
    bool no_output = false;
    float repro_threshold = 150.0f;
    bool quiet = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
            initial_cells = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--frame-output") == 0) {
            frame_output = true;
        } else if (std::strcmp(argv[i], "--no-output") == 0) {
            no_output = true;
        } else if (std::strcmp(argv[i], "--repro-threshold") == 0 && i + 1 < argc) {
            repro_threshold = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--quiet") == 0 || std::strcmp(argv[i], "-q") == 0) {
            quiet = true;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Create environment config
    evo::EnvironmentConfig config = evo::default_environment_config();
    config.width = width;
    config.height = height;
    config.reproduction_threshold = repro_threshold;

    // Create environment
    evo::Environment env(config, seed);
    env.spawn_random_cells(initial_cells, 100.0);

    if (!quiet) {
        std::cout << "Starting simulation:\n"
                  << "  Steps: " << steps << "\n"
                  << "  Initial cells: " << initial_cells << "\n"
                  << "  Grid: " << width << "x" << height << "\n"
                  << "  Seed: " << seed << "\n"
                  << "  Output: " << output_file << "\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (no_output) {
        // No output, just run simulation
        for (int step = 1; step <= steps; step++) {
            env.step();

            if (!quiet && step % 100 == 0) {
                std::cout << "Step " << step << "/" << steps
                          << " - Cells: " << env.cells().size() << "\n";
            }
        }
    } else if (frame_output) {
        // Output individual frame files
        evo::FrameOutputWriter writer(output_file);
        writer.write_frame(env, 0);

        for (int step = 1; step <= steps; step++) {
            env.step();
            writer.write_frame(env, step);

            if (!quiet && step % 100 == 0) {
                std::cout << "Step " << step << "/" << steps
                          << " - Cells: " << env.cells().size() << "\n";
            }
        }
    } else {
        // Output single file with all frames
        evo::OutputWriter writer(output_file);
        writer.write_frame(env);

        for (int step = 1; step <= steps; step++) {
            env.step();
            writer.write_frame(env);

            if (!quiet && step % 100 == 0) {
                std::cout << "Step " << step << "/" << steps
                          << " - Cells: " << env.cells().size() << "\n";
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!quiet) {
        std::cout << "Simulation complete!\n"
                  << "  Final cells: " << env.cells().size() << "\n"
                  << "  Time: " << duration.count() << " ms\n"
                  << "  Speed: " << (steps * 1000.0 / duration.count()) << " steps/sec\n";
    }

    return 0;
}
