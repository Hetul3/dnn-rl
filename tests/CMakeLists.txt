cmake_minimum_required(VERSION 3.10)
project(tiny_rl_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-unused-private-field -Wno-unused-parameter -Wno-unused-variable -Wno-ignored-qualifiers -Wno-delete-non-virtual-dtor")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/../include/tiny_rl/external/tiny-dnn)

# List all test files
set(TEST_SOURCES
    test_envs.cpp
    test_agents.cpp
    test_trainers.cpp
)

# Create individual executable targets for each test file
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
endforeach()

# Create a custom Makefile to handle the "make runtests <filename>" format
file(WRITE ${CMAKE_BINARY_DIR}/Makefile "
# Auto-generated Makefile for running tests

# Default build directory
BUILD_DIR = build

# Default target to create build directory and run cmake
all: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pattern rule to run any test by name
runtests\\: %:
	@if [ ! -f $(BUILD_DIR)/$* ]; then \\
		echo \"Building $*...\"; \\
		cd $(BUILD_DIR) && cmake .. && make $*; \\
	fi
	@cd $(BUILD_DIR) && ./$*

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Help target
help:
	@echo \"Usage:\"
	@echo \"  make             - Build all tests\"
	@echo \"  make runtests test_envs - Build and run test_envs\"
	@echo \"  make clean       - Remove build directory\"

.PHONY: all clean help runtests\\:
")

# Also provide a standard CMake-based approach
add_custom_target(run_all_tests
    COMMENT "Running all tests"
)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_custom_target(run_${test_name}
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        DEPENDS ${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running ${test_name}"
    )
    add_dependencies(run_all_tests ${test_name})
endforeach()