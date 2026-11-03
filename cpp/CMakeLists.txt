cmake_minimum_required(VERSION 3.20)

project(QuantSearch LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib/libtorch")

file(GLOB CUDART_LIB "${CMAKE_SOURCE_DIR}/lib/libtorch/lib/libcudart*.so*")
if(CUDART_LIB)
    if(EXISTS "${CMAKE_SOURCE_DIR}/lib/libtorch/lib/libcudart-8774224f.so.12" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/lib/libtorch/lib/libcudart.so.12")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink libcudart-8774224f.so.12 "${CMAKE_SOURCE_DIR}/lib/libtorch/lib/libcudart.so.12")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink libcudart-8774224f.so.12 "${CMAKE_SOURCE_DIR}/lib/libtorch/lib/libcudart.so")
    endif()
    if(NOT TARGET torch::cudart)
        add_library(torch::cudart INTERFACE IMPORTED)
        set_target_properties(torch::cudart PROPERTIES INTERFACE_LINK_LIBRARIES "${CUDART_LIB}")
    endif()
    set(CAFFE2_USE_CUDA ON)
    set(CUDA_FOUND TRUE)
endif()

find_package(Torch REQUIRED)

add_executable(QuantSearch
        src/main.cpp
        src/environment/trading_env.cpp
        src/environment/portfolio.cpp
        src/features/network_builder.cpp
        src/features/network_feature_extractor.cpp
        src/features/feature_extractor.cpp
        src/agents/baseline_agents.cpp
        src/agents/ppo_matrix_agent.cpp
        src/agents/agent_factory.cpp
        src/agents/experiment_runner.cpp
        src/output/output_files.cpp
)

target_include_directories(QuantSearch PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(QuantSearch PRIVATE
        "${TORCH_LIBRARIES}"
)

set_property(TARGET QuantSearch PROPERTY CXX_STANDARD 20)

if(BUILD_TESTING)
    add_executable(QuantSearchTests
            tests/agent_contract_test.cpp
            src/environment/portfolio.cpp
            src/features/network_builder.cpp
            src/features/network_feature_extractor.cpp
            src/features/feature_extractor.cpp
            src/agents/baseline_agents.cpp
            src/agents/ppo_matrix_agent.cpp
    )
    target_include_directories(QuantSearchTests PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(QuantSearchTests PRIVATE
            "${TORCH_LIBRARIES}"
    )
    add_test(NAME QuantSearchAgentContract COMMAND QuantSearchTests)
endif()