cmake_minimum_required(VERSION 3.20)

project(QuantSearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib/libtorch")

find_package(Torch REQUIRED)

add_executable(QuantSearch
        src/main.cpp
        src/network_builder.cpp
        src/network_feature_extractor.cpp
        src/trading_env.cpp
        src/ppo.cpp
        src/agent.cpp
        src/config.cpp
        src/output_files.cpp
        src/observation.cpp
)

target_include_directories(QuantSearch PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(QuantSearch PRIVATE
        "${TORCH_LIBRARIES}"
)

set_property(TARGET QuantSearch PROPERTY CXX_STANDARD 20)