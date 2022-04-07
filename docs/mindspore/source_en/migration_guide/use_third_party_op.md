# Call Third-Party Operators by Customized Operators

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/use_third_party_op.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When built-in operators cannot meet requirements during network development, you can call the Python API [Custom](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive defined in MindSpore to quickly create different types of customized operators for use.

You can choose different customized operator developing methods base on needs.
See: [custom_operator_custom](https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator_custom.html).

There is a defining method called `aot` which has a special use. The `aot` mode can call the corresponding `cpp`/`cuda` function by loading the precompiled `so`. Therefore, when a third-party library provides the `cpp`/`cuda` function `API`, you can try to call its function interface in `so`.

Here is an example of how to use `Aten` library of PyTorch Aten.

## Using PyTorch Aten operators for Docking

When migrating a network using the PyTorch Aten operator encounters a shortage of built-in operators, we can use the `aot` development method of the `Custom` operator to call PyTorch Aten's operator for fast verification.

PyTorch provides a way to support the introduction of PyTorch's header files to write `cpp/cuda` code by using its associated data structures and compile it into `so`. See:<https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension>.

Using a combination of the two methods, the customized operator can call the PyTorch Aten operator as follows:

### 1. Downloading the Project files

User can download the project files from [here](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/migration_guide/test_custom_pytorch.tar).

Use the following command to extract files and find the folder `test_custom_pytorch`:

```bash
tar xvf test_custom_pytorch.tar
```

The folder include the following files:

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

Using the PyTorch Aten operator focuses mainly on env.sh, setup.py, leaky_relu.cpp/cu, test_*, .py.

Among them, env.sh is used to set environment variables, setup.py is used to compile so, leaky_relu.cpp/cu is used to reference the source code that calls the PyTorch Aten operator, and test_*.py is used to refer to the call Custom operator.

### 2. Writing and calling the Source Code File of PyTorch Aten Operators

Refer to leaky_relu.cpp/cu to write a source code file that calls the PyTorch Aten operator.

The customized operator of `aot` type adopts the `AOT` compilation method, which requires network developers to hand-write the source code file of the operator implementation based on a specific interface, and compile the source code file into a dynamic link library in advance, and then the framework will automatically call the function defined in the dynamic link library. In terms of the development language of the operator implementation, the `GPU` platform supports `CUDA`, and the `CPU` platform supports `C` and `C++`. The interface specification of the operator implemented by the operators in the source code file is as follows:

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);

```

If the `cpu` operator is called, taking `leaky_relu.cpp` as an example, the file provides the function `LeakyRelu` required by `AOT`, which calls `torch::leaky_relu_out`  function of PyTorch Aten:

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
    // If you are using a version without output, the code is as follows:
    // torch::Tensor output = torch::leaky_relu(at_input);
    // at_output.copy_(output);
  return 0;
}

```

If the `gpu` operator is called, take `leaky_relu.cu` as an example:

```cpp
#include <string.h>
#include <torch/extension.h>
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

PyTorch Aten provides operator functions versions with output and operator functions versions without output. Operator functions with output have the '_out' suffix, and PyTorch Aten provides 300+ `apis` of common operators.

When `torch::*_out` is called, `output` copy is not needed. When the versions without `_out`suffix is called, API `torch.Tensor.copy_` is needed to called to result copy.

To see which functions are supported for calling PyTorch Aten, the `CPU` version refers to the PyTorch installation path: `python*/site-packages/torch/include/ATen/CPUFunctions_inl.h` , and for the corresponding `GPU` version, refers to`python*/site-packages/torch/include/ATen/CUDAFunctions_inl.h`。

The apis provided by ms_ext.h are used in the above use case, which are briefly described here:

```cpp
// Convert MindSpore kernel's inputs/outputs to PyTorch Aten's Tensor
std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, c10::Device device) ;
```

### 3. Using the compilation script `setup.py` to generate so

setup.py uses the `cppextension` provided by PyTorch Aten to compile the above `c++/cuda` source code into an `so` file.

Before execution, you need to make sure that PyTorch is installed.

```bash
pip install torch
```

Then add PyTorch's `lib` into `LD_LIBRARY_PATH`。

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
```

Run:

```bash
cpu: python setup.py leaky_relu.cpp leaky_relu_cpu.so
gpu: python setup.py leaky_relu.cu leaky_relu_gpu.so
```

Then the so files that we need may be obtained.

### 4. Using the Customized Operator

Taking CPU as an example, use the Custom operator to call the above PyTorch Aten operator, see the code test_cpu_op.py:

```python
import numpy as np
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops

context.set_context(device_target="CPU")

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
    output = net(Tensor(x0))
    print(output)
```

Run:

```bash
python test_cpu_op.py
```

Result:

```text
[[ 0.    -0.001]
 [-0.002  1.   ]]
```

Attention:

When using a PyTorch Aten `GPU` operator，set `device_target`to `"GPU"`.

```python
context.set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_gpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
```

When using a PyTorch Aten `CPU` operator and `device_target` is `"GPU"`, the settings that need to be added are as follows:

```python
context.set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_cpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
op.add_prim_attr("primitive_target", "CPU")
```

> 1. Compile so with cppextension requires a compiler version that meets the tool's needs, and check for the presence of gcc/clang/nvcc.
> 2. Compile so with cppextension will generate a build folder in the script path, which stores so. The script will copy so to outside of build, but cppextension will skip compilation if it finds that there is already so in build, so if it is a newly compiled so, remember to empty the so under the build.
> 3. The following tests is based on PyTorch 1.9.1，cuda11.1，python3.7. The download link:<https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl>. The cuda version supported by PyTorch Aten needs to be consistent with the local cuda version, and whether other versions are supported needs to be explored by the user.