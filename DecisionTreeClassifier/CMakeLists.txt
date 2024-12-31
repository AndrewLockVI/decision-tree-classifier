
cmake_minimum_required(VERSION 3.10)

# Set project name
project(DecisionTreeClassifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Python and Pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Ensure optimization flag is applied for Release builds (you could also manually override Debug builds)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -fPIC")
else()
    # Optionally you can set optimizations differently for Debug builds
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -Wall -fPIC")
endif()

# Add source files
set(SOURCES
    cpp/DecisionTreeClassifier.cpp
    cpp/TreeNode.cpp
    cpp/Criterion.cpp
    cpp/bindings.cpp
)

# Create the shared library
add_library(decision_tree MODULE ${SOURCES})

# Link with Python and Pybind11
target_include_directories(decision_tree PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(decision_tree PRIVATE ${Python3_LIBRARIES} pybind11::module)

# Set output name based on Python extension suffix
set_target_properties(decision_tree PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Rename the custom clean target to avoid conflict
add_custom_target(clean_build
    COMMAND rm -f ${CMAKE_BINARY_DIR}/*.o
    COMMENT "Clean build files"
)

