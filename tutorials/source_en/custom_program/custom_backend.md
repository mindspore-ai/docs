# Custom Backend

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/custom_backend.md)

## Overview

If the built-in backend is not sufficient to meet the requirements when using MindSpore, you can utilize MindSpore's custom backend feature to enable your own implemented backend.

This tutorial provides a simple custom backend case as a demonstration. For more comprehensive examples, please refer to the [examples](https://gitee.com/mindspore/mindspore/blob/master/tests/st/backend/custom_backend/test_custom_backend.py).

## Implementing Custom Backend

The implementation of custom backend requires completing the following steps:

1. Reference `mindspore/include/custom_backend_api.h` header file.
2. Inherit `BackendBase` class and implement `Build` and `Run` interfaces.
3. Register custom backend by using the `MS_REGISTER_BACKEND` macro.

Based on MindSpore's built-in `ms_backend` backend, additional printing information is added to achieve a simple custom backend.

```c++
#include <string>
#include <memory>
#include <vector>
#include "mindspore/include/custom_backend_api.h"

namespace mindspore {
namespace backend {
constexpr auto kCustomBackendName = "my_custom_backend";

// Use the built-in ms_backend to test the custom backend.
class MSCustomBackendBase : public BackendBase {
 public:
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
    MS_LOG(WARNING) << "MSCustomBackendBase use the origin ms_backend to build the graph.";
    mindspore::backend::BackendManager::GetInstance().Build(func_graph, backend_jit_config, "ms_backend");
  }

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
    MS_LOG(WARNING) << "MSCustomBackendBase use the origin ms_backend to run the graph.";
    mindspore::backend::BackendManager::GetInstance().Run(BackendType::kMsBackend, graph_id, inputs, outputs);
  }
};
MS_REGISTER_BACKEND(kCustomBackendName, MSCustomBackendBase)
}  // namespace backend
}  // namespace mindspore
```

## Compiling Custom Backend

Save the above example code as `custom_backend.cpp` and compile it into `libcustom_backend.so`. The compilation command is as follows:

```cmake
cmake_minimum_required(VERSION 3.16)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use specified MindSpore path
set(MINDSPORE_INCLUDE_DIRS ${MINDSPORE_ROOT}/include)
set(MINDSPORE_LIB_DIRS ${MINDSPORE_ROOT}/lib)
message(STATUS "Using MindSpore from: ${MINDSPORE_ROOT}")

# Build options configuration (simplified)
set(CMAKE_BUILD_TYPE "Release")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Handle MindSpore include directories
if(MINDSPORE_INCLUDE_DIRS)
    include_directories(${include_dir})
    # Add complete MindSpore include paths to ensure all dependency headers are found
    include_directories(${include_dir}/mindspore)
    include_directories(${include_dir}/mindspore/core/include)
    # Add MindSpore ccsrc path, contains mindspore/ccsrc/include/
    include_directories(${include_dir}/mindspore/ccsrc/include)
    # Add MindSpore csrc path, contains mindspore/ccsrc/
    include_directories(${include_dir}/mindspore/ccsrc)
    # Add third_party path, contains securec.h
    include_directories(${include_dir}/third_party)
    include_directories(${include_dir}/third_party/include)
    # Add specific pybind11 path
    include_directories(${include_dir}/third_party/pybind11)
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    if(WIN32)
        if(Python3_DIR)
            message("Python3_DIR set already: " ${Python3_DIR})
        else()
            string(LENGTH ${PYTHON_LIBRARIES} PYTHON_LIBRARIES_LEN)
            string(LENGTH "libpythonxx.a" Python3_NAME_LEN)
            math(EXPR Python3_DIR_LEN  ${PYTHON_LIBRARIES_LEN}-${Python3_NAME_LEN})
            string(SUBSTRING ${Python3_LIBRARIES} 0 ${Python3_DIR_LEN} Python3_DIR)
            message("Python3_DIR: " ${Python3_DIR})
        endif()
        link_directories(${Python3_DIR})
    endif()
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
include_directories(${PYTHON_INCLUDE_DIRS})

# Automatically find all source files
file(GLOB_RECURSE SRC_SOURCES "*.cc")

# Create dynamic library (based on installed MindSpore)
add_library(custom_backend SHARED ${SRC_SOURCES})

# Link MindSpore libraries (based on actual requirements)
target_link_libraries(custom_backend
    ${MINDSPORE_LIB_DIRS}/libmindspore_backend_manager.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_core.so
    ${MINDSPORE_LIB_DIRS}/libmindspore_common.so
)

# ABI flag settings
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation options
target_compile_options(custom_backend PRIVATE
    -fPIC
    -std=c++17
    -Wall
    -Wextra
)

# Installation rules
install(TARGETS custom_backend
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
```

## Using Custom Backend

Using [mindspore.graph.register_custom_backend](https://www.mindspore.cn/docs/en/master/api_python/graph/mindspore.graph.register_custom_backend.html) to register the backend and use [mindspore.jit](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html) to enable the backend:

```python
import mindspore
from mindspore import jit, mint

custom_path = "/data1/libcustom_backend.so"
success = mindspore.graph.register_custom_backend(backend_name="my_custom_backend", path=custom_path)
assert success, "Plugin registration failed"

x = mindspore.Tensor(np.ones([2, 2], np.float32))
y = mindspore.Tensor(np.zeros([2, 2], np.float32))

@jit(backend="my_custom_backend")
def net1(x):
    return mint.sin(x)
x = net1(x)

@jit(backend="ms_backend")
def net2(x):
    return mint.cos(x)
y = net2(y)
```

`net1` will use the `my_custom_backend` custom backend, and `net2` will use the MindSpore built-in `ms_backend` backend.

