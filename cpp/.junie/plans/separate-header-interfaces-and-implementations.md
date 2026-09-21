---
sessionId: session-260916-195234-w7zz
---

# Requirements

### Overview & Goals
The objective is to optimize project structure, compilation efficiency, and codebase maintainability by strictly separating C++ class declarations (interfaces) in `.h` header files from their member function implementations in `.cpp` source files. This provides the highest utility by reducing translation unit bloat, minimizing redundant recompilation cycles, and encapsulating implementation details.

### Scope
- **In Scope**:
  - `../../src/NetworkFeatureExtractor.h` & `src/NetworkFeatureExtractor.cpp`: Extract method bodies to source file; maintain interface in header.
  - `../../src/NetworkBuilder.h` & `src/NetworkBuilder.cpp`: Move `build` and `calculateCorrelationMatrix` definitions to source file; keep declarations and members in header.
  - `../../src/TradingEnv.h` & `src/TradingEnv.cpp`: Move constructor, state machine methods, observation getters, and private helper logic to source file; retain class interface and member variables in header.
  - `include/Printer.h` & `src/Printer.cpp`: Move `printMatrix` and `printLine` static method bodies to source file; keep declarations in header.
  - Necessary header includes and namespace scoping required for compilation.
- **Out of Scope**:
  - Modifying any underlying business logic, algorithms, math formulas, or data structures.
  - Adding new features or altering existing public API signatures.
  - Modifying `../../src/ConvolutionalFeatureExtractor.h`, `include/PriceFeatureExtractor.h`, or `../../src/VolumeFeatureExtractor.h` which already contain pure interface/class declarations.

### Functional Requirements
- **FR-1**: All method implementations currently embedded within `.h` files must be relocated to their corresponding `.cpp` translation units.
- **FR-2**: All `.h` header files must define clean class declarations containing only member function prototypes, class constants, and member variables.
- **FR-3**: No behavioral changes or algorithmic alterations shall be introduced to any functions during migration.
- **FR-4**: Target executable `QuantSearch` must continue to build and link without compilation or linkage errors.

# Technical Design

### Current Implementation
Currently, several header files contain inline function bodies:
- `../../src/NetworkFeatureExtractor.h`: Contains stub definitions for `extractNodeLevelFeatures` and `extractLinkLevelFeatures`.
- `../../src/NetworkBuilder.h`: Contains complete implementations for `build` and `calculateCorrelationMatrix`.
- `../../src/TradingEnv.h`: Contains full definitions for constructor, environment transitions (`step`, `reset`), getters, file I/O (`loadData`), and mathematical utility functions.
- `include/Printer.h`: Contains full inline definitions for `printMatrix` and `printLine`.

The corresponding `.cpp` files in `src/` currently only contain boilerplate include statements (`#include "..."`).

### Key Decisions
- **Strict Declaration/Implementation Separation**: Maximize modularity and build efficiency by placing only declarations in headers and function definitions in `.cpp` files.
- **Maintain Method Signatures and Attributes**: Retain attributes such as `[[nodiscard]]`, `const` qualifiers, default member initializers, and access specifiers (`public` / `private`) identically to preserve binary interface integrity and caller expectations.
- **Include Granularity**: Include heavy standard library headers (e.g., `<fstream>`, `<sstream>`, `<iomanip>`, `<algorithm>`) directly in `.cpp` files where implementations require them, keeping headers lightweight.

### Proposed Changes

#### 1. `NetworkFeatureExtractor` (`../../src/NetworkFeatureExtractor.h` & `src/NetworkFeatureExtractor.cpp`)
- **Header**:
  ```cpp
  #pragma once
  #ifndef QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H
  #define QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H

  #include <vector>

  class NetworkFeatureExtractor {
  public:
      // [stock1 -> [Node Degree, Weighted Node Degree, Average Neighbor Degree, Propensity of increase degree, Node Betweenes, Node Closeness, Node Eigenvector, Node Clustering Coefficient]]
      std::vector<std::vector<double>> extractNodeLevelFeatures(std::vector<std::vector<double>> graph);

      // [stock1 -> [stock2 -> [Link Existence, Correlation Value, Common neighbors, Jaccard Coefficient, Adamic-Adar Coefficient, Sorenson-Dice Coefficient, Edge Betweenness, Same Community, Preferential Attachment ]]]
      std::vector<std::vector<std::vector<double>>> extractLinkLevelFeatures(std::vector<std::vector<double>> graph);
  };

  #endif // QUANTSEARCH_NETWORKFEATUREEXTRACTOR_H
  ```
- **Source**: Implement `NetworkFeatureExtractor::extractNodeLevelFeatures` and `NetworkFeatureExtractor::extractLinkLevelFeatures`.

#### 2. `NetworkBuilder` (`../../src/NetworkBuilder.h` & `src/NetworkBuilder.cpp`)
- **Header**:
  ```cpp
  #pragma once
  #ifndef QUANTSEARCH_NETWORKBUILDER_H
  #define QUANTSEARCH_NETWORKBUILDER_H

  #include <vector>

  class NetworkBuilder {
      float base_threshold = 0.5;

  public:
      std::vector<std::vector<double>> build(const std::vector<std::vector<double>>& obs);

  private:
      std::vector<std::vector<double>> calculateCorrelationMatrix(const std::vector<std::vector<double>>& obs);
  };

  #endif // QUANTSEARCH_NETWORKBUILDER_H
  ```
- **Source**: Implement `NetworkBuilder::build` and `NetworkBuilder::calculateCorrelationMatrix` with required `<algorithm>` and `<cmath>` includes.

#### 3. `TradingEnv` (`../../src/TradingEnv.h` & `src/TradingEnv.cpp`)
- **Header**:
  ```cpp
  #ifndef QUANTSEARCH_TRADINGENV_H
  #define QUANTSEARCH_TRADINGENV_H

  #include <vector>
  #include <string>

  class TradingEnv {
  public:
      TradingEnv(const float train_percent);
      void reset();
      double step(const std::vector<double>& action);

      [[nodiscard]] std::vector<std::vector<double>> getPricesObservation() const;
      [[nodiscard]] std::vector<std::vector<double>> getVolumeObservation() const;
      [[nodiscard]] std::vector<double> getPercentPortfolio() const;
      [[nodiscard]] bool terminated() const;
      [[nodiscard]] bool train_terminated() const;

  private:
      int CLOSE_COL = 1;
      int VOLUME_COL = 5;
      int OBS_LEN = 50;
      std::string DATA_DIR = "../data";

      int stocks_amount = 0;
      int current_step = 0;
      unsigned long total_of_days = 0;
      int test_days = 0;
      double initial_cash = 10000;
      std::vector<double> cash_portfolio;
      std::vector<double> stock_portfolio;
      std::vector<double> percent_portfolio;
      std::vector<std::vector<double>> closes;
      std::vector<std::vector<double>> volumes;

      void loadData();
      [[nodiscard]] std::vector<double> calculateInitialCashPortfolio() const;
      [[nodiscard]] std::vector<double> calculateCashPortfolioByStockPortfolio() const;
      [[nodiscard]] std::vector<double> calculateCashPortfolioByPercentPortfolio() const;
      [[nodiscard]] std::vector<double> calculateStockPortfolioByCashPortfolio() const;
      [[nodiscard]] std::vector<double> calculatePercentPortfolioByCashPortfolio() const;
      [[nodiscard]] double calculateTotalCash() const;
  };

  #endif // QUANTSEARCH_TRADINGENV_H
  ```
- **Source**: Implement all member methods in `src/TradingEnv.cpp` with `<fstream>`, `<sstream>`, `<filesystem>`, and `<cmath>`.

#### 4. `Printer` (`include/Printer.h` & `src/Printer.cpp`)
- **Header**:
  ```cpp
  #ifndef QUANTSEARCH_PRINTER_H
  #define QUANTSEARCH_PRINTER_H

  #include <vector>

  class Printer {
  public:
      static void printMatrix(const std::vector<std::vector<double>>& matrix);
      static void printLine();
  };

  #endif // QUANTSEARCH_PRINTER_H
  ```
- **Source**: Implement `Printer::printMatrix` and `Printer::printLine` with `<iostream>` and `<iomanip>`.

### File Structure Changes
- `../../src/NetworkFeatureExtractor.h` (modified — declaration only)
- `src/NetworkFeatureExtractor.cpp` (modified — implementations added)
- `../../src/NetworkBuilder.h` (modified — declaration only)
- `src/NetworkBuilder.cpp` (modified — implementations added)
- `../../src/TradingEnv.h` (modified — declaration only)
- `src/TradingEnv.cpp` (modified — implementations added)
- `include/Printer.h` (modified — declaration only)
- `src/Printer.cpp` (modified — implementations added)

# Testing

### Validation Approach
Verification focuses on static structure validation, header independence, compilation integrity, and linking consistency across the CMake build targets.

### Key Scenarios
- **Compilation Verification**: Build target `QuantSearch` using CMake/Make to verify that all translation units compile cleanly without unresolved symbols, duplicate definitions (`ODR` violations), or missing declarations.
- **Interface Completeness**: Verify that `src/main.cpp` can include all headers and call methods without encountering missing symbol or visibility errors.
- **Zero Behavior Modification**: Ensure no logic or algorithm branches were altered during relocation of method bodies.

### Edge Cases
- Ensuring `[[nodiscard]]` and `const` qualifiers match strictly between header declarations and `.cpp` definitions.
- Ensuring static member functions in `Printer` are defined without the `static` keyword in the `.cpp` file according to C++ syntax rules.
- Proper standard library header includes in `.cpp` files (`<cmath>`, `<algorithm>`, `<fstream>`, `<sstream>`, `<filesystem>`, `<iostream>`, `<iomanip>`).

# Delivery Steps

### ✓ Step 1: Refactor NetworkFeatureExtractor and NetworkBuilder into header interfaces and source implementations
`NetworkFeatureExtractor` and `NetworkBuilder` have clean header interfaces with implementations isolated in their respective `.cpp` files.

- Update `../../src/NetworkFeatureExtractor.h` to retain member declarations (`extractNodeLevelFeatures`, `extractLinkLevelFeatures`) under `public:` accessibility and preserve method doc comments.
- Move the method bodies of `extractNodeLevelFeatures` and `extractLinkLevelFeatures` into `src/NetworkFeatureExtractor.cpp` using qualified member syntax `NetworkFeatureExtractor::...`.
- Update `../../src/NetworkBuilder.h` to declare `build` and `calculateCorrelationMatrix` while retaining the member variable `base_threshold`.
- Move the complete implementations of `build` and `calculateCorrelationMatrix` into `src/NetworkBuilder.cpp` with necessary standard library includes (`<vector>`, `<cmath>`, `<algorithm>`).

### ✓ Step 2: Refactor TradingEnv and Printer into header interfaces and source implementations
`TradingEnv` and `Printer` have pure declaration interfaces in header files with all execution logic migrated into `.cpp` files.

- Update `../../src/TradingEnv.h` to declare the constructor, public member methods (`reset`, `step`, `getPricesObservation`, `getVolumeObservation`, `getPercentPortfolio`, `terminated`, `train_terminated`), and private helper methods (`loadData`, `calculateInitialCashPortfolio`, etc.), keeping all member variable definitions in the header.
- Move the implementation of all `TradingEnv` methods into `src/TradingEnv.cpp` with necessary standard library includes (`<fstream>`, `<sstream>`, `<cmath>`, `<filesystem>`, `<vector>`, `<string>`).
- Update `include/Printer.h` to declare static methods `printMatrix` and `printLine`.
- Implement `Printer::printMatrix` and `Printer::printLine` in `src/Printer.cpp` including `<iostream>` and `<iomanip>`.

### * Step 3: Validate header hygiene and compile the project
The entire project compiles cleanly with LibTorch and all translation units link properly against the updated headers.

- Validate header guard consistency (`#pragma once` and `#ifndef` / `#define` blocks) across all modified header files.
- Ensure standard namespace hygiene and appropriate `#include` directives are contained within `.cpp` files rather than leaking across translation units.
- Verify that `QuantSearch` builds without compilation or linker errors.