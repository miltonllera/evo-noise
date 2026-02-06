CXX = clang++
CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic
LDFLAGS =

# Source files
SRCS = $(wildcard src/*.cpp)
OBJS = $(SRCS:src/%.cpp=build/%.o)
TARGET = evo_noise

# Debug build
DEBUG_CXXFLAGS = -std=c++20 -g -O0 -Wall -Wextra -Wpedantic -fsanitize=address
DEBUG_LDFLAGS = -fsanitize=address
DEBUG_OBJS = $(SRCS:src/%.cpp=build/debug/%.o)
DEBUG_TARGET = evo_noise_debug

.PHONY: all clean debug run benchmark

all: $(TARGET)

debug: $(DEBUG_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CXX) $(DEBUG_LDFLAGS) -o $@ $^

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -Iinclude -c -o $@ $<

build/debug/%.o: src/%.cpp | build/debug
	$(CXX) $(DEBUG_CXXFLAGS) -Iinclude -c -o $@ $<

build:
	mkdir -p build

build/debug:
	mkdir -p build/debug

clean:
	rm -rf build $(TARGET) $(DEBUG_TARGET)

run: $(TARGET)
	./$(TARGET) --steps 1000 --cells 100 --seed 42

benchmark: $(TARGET)
	@echo "Running benchmark..."
	@./$(TARGET) --steps 1000 --cells 100 --seed 42 --quiet
	@echo ""
	@echo "Running with more cells..."
	@./$(TARGET) --steps 1000 --cells 500 --seed 42 --quiet
