# Custom Pass

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/custom_pass.md)

## Overview

When it is necessary to change the structure of the computation graph, you can utilize MindSpore's custom pass feature to write pass logic, implement and register a custom pass plugin, and optimize the structure of the computation graph.

This tutorial provides a simple custom pass case as a demonstration. For more comprehensive examples, please refer to the [examples](https://gitee.com/mindspore/mindspore/blob/master/tests/st/backend/custom_pass/test_custom_passes.py).

## Implementing Custom Pass

The implementation of custom pass requires completing the following steps:

1. Reference `mindspore/include/custom_pass_api.h` header file.
2. Inherit `PatternToPatternPass` class and implement `DefineSrcPattern`, `DefineDstPattern` and `CheckMatchedDAG` interfaces.
3. Inherit `CustomPassPlugin` class and implement `GetPluginName`, `GetAvailablePassNames` and `CreatePass` interfaces.
4. Register custom backend by using the `EXPORT_CUSTOM_PASS_PLUGIN` macro.

Here, we implement a simple AddNegFusionPass and a custom Pass plugin to replace the Add operator and Neg operator with a Sub operator.

```c++
// add_neg_fusion_pass.h
// header file of AddNegFusionPass
#ifndef MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_
#define MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_

#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {
/**
 * @brief Pass to fuse Add and Neg operations into Sub
 *
 * Transforms Add(x, Neg(y)) into Sub(x, y)
 * This is a standard algebraic optimization that eliminates unnecessary Neg operations
 * Works on CPU/GPU/Ascend since all platforms support Add, Neg, and Sub operations
 * Inherits from PatternToPatternPass to comply with MindSpore plugin system requirements
 */
class AddNegFusionPass : public PatternToPatternPass {
 public:
  AddNegFusionPass() : PatternToPatternPass("AddNegFusionPass") {}

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                       const AnfNodePtr &node) const override;

 private:
  static bool IsAddNode(const AnfNodePtr &node);
  static bool IsNegNode(const AnfNodePtr &node);

  static AnfNodePtr BuildSub(const PatternMap &m, const AnfNodePtr &default_node);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CUSTOM_PASS_ADD_NEG_FUSION_PASS_H_
```

```c++
// add_neg_fusion_pass.cc
// cpp file of AddNegFusionPass
#include "add_neg_fusion_pass.h"

namespace mindspore {
namespace opt {
void AddNegFusionPass::DefineSrcPattern(SrcPattern *src_pattern) {
  MS_LOG(INFO) << "Defining source pattern for AddNegFusionPass";
  MS_EXCEPTION_IF_NULL(src_pattern);

  // Pattern: Add(x, Neg(y))
  (*src_pattern)
    .AddVar("x")
    .AddVar("y")
    .AddCNode("neg", {std::make_shared<Primitive>("Neg"), "y"})
    .AddCNode("add", {std::make_shared<Primitive>("Add"), "x", "neg"});

  MS_LOG(INFO) << "Source pattern defined: Add(x, Neg(y))";
}

AnfNodePtr AddNegFusionPass::BuildSub(const PatternMap &m, const AnfNodePtr &default_node) {
  auto add_node = m.Get("add")->cast<CNodePtr>();
  auto neg_node = m.Get("neg")->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_node);
  MS_EXCEPTION_IF_NULL(neg_node);

  auto sub_node = default_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sub_node);

  // Copy Add node's scope to maintain execution context
  sub_node->set_scope(add_node->scope());

  // Set abstract same as Add output
  auto add_abstract = add_node->abstract();
  if (add_abstract != nullptr) {
    sub_node->set_abstract(add_abstract->Clone());
  } else {
    MS_LOG(EXCEPTION) << "Failed to create Sub abstract from Add node";
  }

  return sub_node;
}

void AddNegFusionPass::DefineDstPattern(DstPattern *dst_pattern) {
  MS_LOG(INFO) << "Defining destination pattern for AddNegFusionPass";
  MS_EXCEPTION_IF_NULL(dst_pattern);

  // Replace with Sub(x, y) - directly subtract y instead of adding its negation
  (*dst_pattern).AddCNode("sub", {std::make_shared<Primitive>("Sub"), "x", "y"}, BuildSub);

  MS_LOG(INFO) << "Destination pattern defined: Sub(x, y)";
}

bool AddNegFusionPass::CheckMatchedDAG(const PatternMap &pattern_map, const FuncGraphPtr &func_graph,
                                       const AnfNodePtr &node) const {
  auto add_node = pattern_map.Get("add");
  if (!add_node) {
    MS_LOG(ERROR) << "Add node not found in pattern match";
    return false;
  }

  auto neg_node = pattern_map.Get("neg");
  if (!neg_node) {
    MS_LOG(ERROR) << "Neg node not found in pattern match";
    return false;
  }

  auto x_node = pattern_map.Get("x");
  if (!x_node) {
    MS_LOG(ERROR) << "x node not found in pattern match";
    return false;
  }

  auto y_node = pattern_map.Get("y");
  if (!y_node) {
    MS_LOG(ERROR) << "y node not found in pattern match";
    return false;
  }

  MS_LOG(INFO) << "AddNeg fusion pattern matched successfully";
  return true;
}
}  // namespace opt
}  // namespace mindspore
```

```c++
// ms_custom_pass_plugin.cc
// cpp file of Custom Pass Plugin
#include <string>
#include <memory>
#include <vector>
#include "mindspore/ccsrc/include/backend/common/custom_pass/custom_pass_plugin.h"
#include "add_neg_fusion_pass.h"

namespace mindspore {
namespace opt {

class MSCustomPassPlugin : public CustomPassPlugin {
 public:
  std::string GetPluginName() const override { return "ms_custom_pass_plugin"; }

  std::vector<std::string> GetAvailablePassNames() const override {
    return {"ReplaceAddNFusionPass", "AddNegFusionPass"};
  }

  std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const override {
    if (pass_name == "AddNegFusionPass") {
      auto pass = std::make_shared<AddNegFusionPass>();
      MS_LOG(INFO) << "Created pass '" << pass_name << "' successfully";
      return pass;
    } else {
      MS_LOG(WARNING) << "Pass '" << pass_name << "' not found, available: ReplaceAddNFusionPass, AddNegFusionPass";
      return nullptr;
    }
  }
};
}  // namespace opt
}  // namespace mindspore

EXPORT_CUSTOM_PASS_PLUGIN(mindspore::opt::MSCustomPassPlugin)
```

## Compiling Custom Pass Plugin

Compile the above example code into `libcustom_pass.so`. The CMake script is as follows:

```cmake
cmake_minimum_required(VERSION 3.16)
project(pass VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use specified MindSpore path
set(MINDSPORE_INCLUDE_DIR ${MINDSPORE_ROOT}/include)
set(MINDSPORE_LIB_DIRS ${MINDSPORE_ROOT}/lib)
message(STATUS "Using MindSpore from: ${MINDSPORE_ROOT}")

# Build options configuration (simplified)
set(CMAKE_BUILD_TYPE "Release")

# Set CMake module path - adjusted for mindspore test location
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Handle multiple MindSpore include directories
if(MINDSPORE_INCLUDE_DIR)
    # Add complete MindSpore include paths to ensure all dependency headers are found
    include_directories(${MINDSPORE_INCLUDE_DIR})
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore)
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/core/include)
    # Add MindSpore ccsrc path
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/ccsrc/)
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/ccsrc/include/)
    # Add MindSpore ops path
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/ops)
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/ops/include)
    include_directories(${MINDSPORE_INCLUDE_DIR}/mindspore/ops/kernel/include)
    # Add third_party path, contains securec.h
    include_directories(${MINDSPORE_INCLUDE_DIR}/third_party)
    # Add specific securec path
    include_directories(${MINDSPORE_INCLUDE_DIR}/third_party/securec/include)
endif()

# Automatically find all source files
file(GLOB_RECURSE PASS_SOURCES "*.cc")
file(GLOB_RECURSE PASS_HEADERS "*.h")

# Create dynamic library (based on installed MindSpore)
add_library(custom_pass SHARED ${PASS_SOURCES})

# Link MindSpore libraries (based on actual requirements)
target_link_libraries(custom_pass
    ${MINDSPORE_LIB_DIRS}/libmindspore_backend_common.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_core.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_common.so
)

# Default settings
option(ENABLE_GLIBCXX "enable_glibcxx" OFF)

# System-related overrides
if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ENABLE_GLIBCXX ON)
endif()

# Environment variable overrides
if(DEFINED ENV{ENABLE_GLIBCXX})
    set(ENABLE_GLIBCXX $ENV{ENABLE_GLIBCXX})
endif()

# ABI flag settings
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation options
target_compile_options(custom_pass PRIVATE
    -fPIC
    -std=c++17
    -Wall
    -Wextra
)

# Use ABI settings consistent with MindSpore
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        target_compile_definitions(custom_pass PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation definitions
target_compile_definitions(custom_pass PRIVATE
    -DPASS_PLUGIN_EXPORTS
    -DMINDSPORE_PASS
)

# Set dynamic library properties
set_target_properties(custom_pass PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PREFIX "lib"
    OUTPUT_NAME "custom_pass"
)

# Installation rules
install(TARGETS custom_pass
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
```

The compilation command is as follows:

```bash
cmake . -DMINDSPORE_ROOT=/path/to/mindspore
make
```

`/path/to/mindspore` represents the installation path of MindSpore.

## Using Custom Pass

Using [mindspore.graph.register_custom_pass](https://www.mindspore.cn/docs/en/master/api_python/graph/mindspore.graph.register_custom_pass.html) to register and enable the custom pass:

```python
import numpy as np
import mindspore
from mindspore import jit, ops, nn, context, Tensor

custom_path = "/data1/libcustom_pass.so"
success = mindspore.graph.register_custom_pass("AddNegFusionPass", custom_path, "cpu")
assert success, "Plugin registration failed"

class AddNegNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.neg = ops.Neg()

    @jit(backend="ms_backend")
    def construct(self, x1, x2):
        # Neg operation: -x2
        neg_x2 = self.neg(x2)
        # Add operation: x1 + (-x2) = x1 - x2
        output = x1 + neg_x2
        return output

context.set_context(device_target="CPU")
net = AddNegNetwork()
x1 = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]).astype(np.float32))
x2 = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                        [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]]).astype(np.float32))
output = net(x1, x2)

# Verify functional correctness
expected = x1.asnumpy() - x2.asnumpy()  # x1 + (-x2) = x1 - x2
np.testing.assert_array_almost_equal(output.asnumpy(), expected)
```
