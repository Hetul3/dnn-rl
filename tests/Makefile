CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra \
    -I../include -I../include/tiny_rl/external/tiny-dnn/ \
    -Wno-deprecated-copy \
    -Wno-deprecated-declarations \
    -Wno-unused-private-field \
    -Wno-unused-parameter \
    -Wno-unused-variable \
    -Wno-ignored-qualifiers \
    -Wno-delete-non-virtual-dtor

# List all test source files
TEST_SOURCES = test_envs.cpp test_agents.cpp test_trainers.cpp
TEST_EXECUTABLES = $(TEST_SOURCES:.cpp=)

# Default target builds all tests
all: $(TEST_EXECUTABLES)

# Pattern rule for building test executables
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(TEST_EXECUTABLES)

# Target to run a specific test
runtests:
	@if [ -z "$(test)" ]; then \
		echo "Error: No test specified. Usage: make runtests test=<test_name>"; \
		exit 1; \
	fi; \
	if [ ! -f "$(test)" ] && [ -f "$(test).cpp" ]; then \
		echo "Building $(test)..."; \
		$(MAKE) $(test); \
	elif [ ! -f "$(test)" ] && [ ! -f "$(test).cpp" ]; then \
		echo "Error: Test '$(test)' not found"; \
		exit 1; \
	fi; \
	echo "Running $(test)..."; \
	./$(test); \
	rm -f $(test)

# Helper target to run all tests
runall: all
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test..."; \
		./$$test; \
		rm -f $$test; \
	done

# Help target
help:
	@echo "Usage:"
	@echo "  make              - Build all tests"
	@echo "  make runtests test=test_envs - Build and run the test_envs test"
	@echo "  make runall       - Run all tests"
	@echo "  make clean        - Remove test executables"

.PHONY: all clean runall help runtests