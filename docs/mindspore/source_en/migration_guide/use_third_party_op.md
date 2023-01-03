# Using Third-party Operator Libraries Based on Customized Interfaces

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/migration_guide/use_third_party_op.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When lacking of the built-in operators during developing a network, you can use the primitive in [Custom](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) to easily and quickly define and use different types of customized operators.

Developers can choose different customized operator development methods according to their needs. For details, please refer to the [Usage Guide](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/operation/op_custom.html) of Custom operator.

One of the development methods for customized operators, the `aot` method, has its own special use. The `aot` can call the corresponding `cpp`/`cuda` functions by loading a pre-compiled `so`. Therefore. When a third-party library provides `API`, a `cpp`/`cuda` function, you can try to call its function interface in `so`, which is described below by taking `Aten` library in PyTorch as an example.

## PyTorch Aten Operator Matching

When lacking of built-in operators during migrating a network that uses the PyTorch Aten operator, we can use the `aot` development of the `Custom` operator to call the PyTorch Aten operator for fast verification.

PyTorch provides a way to support the introduction of PyTorch header files, so that `cpp/cuda` code can be written by using related data structures and compiled into `so`. Reference: <https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension>.

Using a combination of the two approaches, the customized operator can call the PyTorch Aten operator, which is used as follows:

### 1. Downloading Project Files

The project files can be downloaded [here](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/migration_guide/test_custom_pytorch.tar).

Use the following command to extract the zip package and get the folder `test_custom_pytorch`.

```bash
tar xvf test_custom_pytorch.tar
```

The folder contains the following files:

```text
test_custom_pytorch
├── env.sh                           # set PyTorch/lib into LD_LIBRARY_PATH
├── leaky_relu.cpp                   # an example of use Aten CPU operator
├── leaky_relu.cu                    # an example of use Aten GPU operator
├── ms_ext.cpp                       # convert Tensors between MindSpore and PyTorch
├── ms_ext.h                         # convert API
├── README.md
├── run_cpu.sh                       # a script to run cpu case
├── run_gpu.sh                       # a script to run gpu case
├── setup.py                         # a script to compile cpp/cu into so
├── test_cpu_op_in_gpu_device.py     # a test file to run Aten CPU operator on GPU device
├── test_cpu_op.py                   # a test file to run Aten CPU operator on CPU device
└── test_gpu_op.py                   # a test file to run Aten GPU operator on GPU device
```

Using PyTorch Aten arithmetic focuses on env.sh, setup.py, leaky_relu.cpp/cu, and test_*.py.

Among them, env.sh is used to set environment variables, setup.py is used to compile so, leaky_relu.cpp/cu is a reference for used to write the source code for calling PyTorch Aten operator in reference, and test_*.py is used to call Custom operator in reference.

### 2. Writing Source Code File that Calls PyTorch Aten Operator

Refer to leaky_relu.cpp/cu to write the source code file that calls the PyTorch Aten operator.

Since customized operators of type `aot` are compiled by using `AOT`, the developer is required to write the source code files corresponding to the operator implementation functions based on a specific interface and compile the source code files into a dynamic link library in advance, and the framework will automatically call the functions in the dynamic link library for execution during the web runtime. As for the development language of the operator implementation, `CUDA` is supported for `GPU` platform, and `C` and `C++` are supported for `CPU` platform. The interface specification of the operator implementation functions in the source code file is as follows.

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);

```

If `cpu` operator is called, taking `leaky_relu.cpp` as an example, the file provides `LeakyRelu` required by `AOT`, which calls `torch::leaky_relu_out` of PyTorch Aten:

```cpp
#include <string.h>
#include <torch/extension.h> // Header file reference section
#include "ms_ext.h"

extern "C" int LeakyRelu(
    int nparam,
    void** params,
    int* ndims,
    int64_t** shapes,
    const char** dtypes,
    void* stream,
    void* extra) {
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
    auto at_input = tensors[0];
    auto at_output = tensors[1];
    torch::leaky_relu_out(at_output, at_input);
    // If you use the version without output, the code is as follows:
    // torch::Tensor output = torch::leaky_relu(at_input);
    // at_output.copy_(output);
  return 0;
}

```

If `gpu` operator is called, taking `leaky_relu.cu` as an example:

```cpp
#include <string.h>
#include <torch/extension.h> // Header file reference section
#include "ms_ext.h"

extern "C" int LeakyRelu(
    int nparam,
    void** params,
    int* ndims,
    int64_t** shapes,
    const char** dtypes,
    void* stream,
    void* extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(custream);
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
    auto at_input = tensors[0];
    auto at_output = tensors[1];
    torch::leaky_relu_out(at_output, at_input);
  return 0;
}
```

Among them, PyTorch Aten provides a version of the operator function with output and a version of the operator function without output. The operator function with output has the `_out` suffix, and PyTorch Aten provides `api` for 300+ commonly-used operators.

When calling `torch::*_out`, `output` copy is not required. When calling the version without the `_out` suffix, calling the API `torch.Tensor.copy_` to make a copy of the result is required.

To see which functions of PyTorch Aten are supported, refer to the PyTorch installation path: `python*/psite-packages/torch/include/ATen/CPUFunctions_inl.h` for the `CPU` version and `python*/ site-packages/torch/include/ATen/CUDAFunctions_inl.h` for the `GPU` version.

The above use case uses the api provided by ms_ext.h, which is described here:

```cpp
// Transform inputs/outputs of MindSpore kernel as Tensor of PyTorch Aten
std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, c10::Device device) ;
```

### 3. Using the Compile Script `setup.py` to Generate so

setup.py compiles the above `c++/cuda` source code into a `so` file by using the `cppextension` provided by PyTorch Aten.

You need to make sure PyTorch is installed before executing it.

```bash
pip install torch
```

Add `lib` of PyTorch to `LD_LIBRARY_PATH`.

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
```

Execute:

```bash
cpu: python setup.py leaky_relu.cpp leaky_relu_cpu.so
gpu: python setup.py leaky_relu.cu leaky_relu_gpu.so
```

Get so file we need.

### 4. Using Customized Operators

Taking the CPU as an example, the above PyTorch Aten operator is called by using the Custom operator. The code can be found in test_cpu_op.py:

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(device_target="CPU")

def LeakyRelu():
    return ops.Custom("./leaky_relu_cpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.leaky_relu = LeakyRelu()

    def construct(self, x):
        return self.leaky_relu(x)

if __name__ == "__main__":
    x0 = np.array([[0.0, -0.1], [-0.2, 1.0]]).astype(np.float32)
    net = Net()
    output = net(ms.Tensor(x0))
    print(output)
```

Execute:

```bash
python test_cpu_op.py
```

The result is:

```text
[[ 0.    -0.001]
 [-0.002  1.   ]]
```

Note:

If you are using the PyTorch Aten `GPU` operator, `device_target` needs to be set to `"GPU"`.

```python
set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_gpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
```

If the PyTorch Aten `CPU` operator is used and the `device_target` is set to `"GPU"`, you need to add the following settings:

```python
set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_cpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
op.set_device("CPU")
```

> 1. To compile so with cppextension, you need to meet the compiler version required by the tool and check if gcc/clang/nvcc exists.
> 2. Using cppextension to compile so will generate a build folder in the script path, which stores so. The script will copy so outside build, but cppextension will skip the compilation if it finds that there is already so in the build, so remember to clear newly-compiled so under build.
> 3. The above tests are based on [PyTorch 1.9.1，cuda11.1，python3.7](https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl). The cuda version supported by PyTorch Aten should be the same as the local cuda version, and whether other versions is supported should be explored by the user.
