# QuantSearch C++

## Layout

- `src/main.cpp` starts the configured experiment run; `src/config.hpp` contains the run and PPO settings.
- `src/environment/` loads the market CSVs and exposes close/volume observations as ordinary C++ matrices. Portfolio accounting is separate from the market environment.
- `src/features/` transforms those matrices into named vectors, matrices, and 3D matrices.
- `src/agents/` defines the abstract `Agent::act` contract, baseline agents, trainable agents, the factory, and the experiment runner. LibTorch is used only by the PPO agent implementation here.
- `src/output/` writes one row per market day, with money and allocation weights for every configured agent.

## Choosing and adding agents

Edit `Config::ENABLED_AGENTS` in `src/config.hpp` to choose the agents for a run. Supported names are `price_matrix`, `volume_matrix`, `dual_matrix`, `convolutional` (both price and volume networks), `convolutional_price` (price network only), `convolutional_volume` (volume network only), `same_percent`, and `best_performing`. The first three PPO agents use the raw close-history matrix, raw volume-history matrix, or both, respectively; the convolutional agents use the derived network correlation matrices. The checked-in configuration compares the convolutional agents against both baselines (`same_percent` and `best_performing`).

New agents derive from `Agent` and implement `name()` and `act(const AgentInput&)`. `AgentInput` is a map of feature names to C++ `Vector`, `Matrix`, or `Matrix3D` values. Feature extraction and agent actions use no LibTorch types. An agent that learns from rewards can derive from `TrainableAgent` and implement its episode/update hooks; all tensor handling stays inside its implementation under `src/agents/`.

The training/test day counts, episode count, initial money, thresholds, PPO options, and enabled agent list are configured in `src/config.hpp`.

## Build and test

From this directory:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the executable with the working directory set to this project or one of its build directories; it locates `_data`, `_logs`, and `_results` from the current directory or its parents.
