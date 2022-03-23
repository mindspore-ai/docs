# Use Third-Party Operators by Custom Operators

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/use_third_party_op.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

When built-in operators cannot meet requirements during network development, you can call the Python API [Custom](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom) primitive defined in MindSpore to quickly create different types of custom operators for use.

You can choose different custom operator defining methods base on needs.
See: [custom_operator_custom](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_custom.html).

There is a defining method called `aot` which has a special use. It could load a dynamic library and use cpp/cuda functions in it. When third-party library provide some API, we could try to use these APIs in the dynamic library.

Here is an example of how to use PyTorch Aten by Custom operator.

## Use PyTorch Aten operators by Custom operator

When migrate a network script which using PyTorch Aten ops, we could use `Custom` operator to reuse Pytorch Aten operators if Mindspore missing the operator.

PyTorch provides a mechanism called C++ extensions that allow users to create PyTorch operators defined out-of-source. It makes user easy to use Pytorch data structure to write cpp/cuda code, and compile it into dynamic library. See:<https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension>.

So Custom operator can use this mechanism to call PyTorch Aten operators. Here is an example of usage:

### 1. Download the Project files

User can download the project files from [here](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/migration_guide/test_custom_pytorch.tar).

Use `tar` to extract files into folder `test_custom_pytorch`:

```bash
tar xvf test_custom_pytorch.tar
```

Folder `test_custom_pytorch`include files:

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

### 2. Write a CPP/CU file to use PyTorch Aten operators

The custom operator of aot type adopts the AOT compilation method, which requires network developers to hand-write the source code file of the operator implementation based on a specific interface, and compile the source code file into a dynamic library in advance, and then the framework will automatically call and run the function defined in the dynamic library. In terms of the development language of the operator implementation, the GPU platform supports CUDA, and the CPU platform supports C and C++. The interface specification of the operator implementation in the source file is as follows:

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);

```

Take leaky_relu.cpp as an example for `cpu` backend:

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
    auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
    auto at_input = tensors[0];
    auto at_output = tensors[1];
    torch::leaky_relu_out(at_output, at_input);
    // a case which need copy output
    // torch::Tensor output = torch::leaky_relu(at_input);
    // at_output.copy_(output);
  return 0;
}

```

Take leaky_relu.cu as an example for `gpu` backend:

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

PyTorch Aten provides 300+ operator APIs with/without output tensors.

`torch::*_out` is with output tensors which do not need memory copy.

The APIs Without output tensors need use `torch.Tensor.copy_` to copy return value to kernel output.

For more details of APIs, see: `python*/site-packages/torch/include/ATen/CPUFunctions_inl.h` and `python*/site-packages/torch/include/ATen/CUDAFunctions_inl.h`.

A brief introduction of project APIs in ms_ext.h:

```cpp
// Convert MindSpore kernel's inputs/outputs to PyTorch Aten's Tensor
std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, c10::Device device) ;
```

### 3. Use `setup.py` to compile source code into dynamic library

Install Pytorch first.

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

Then the needed dynamic library will be created.

### 4. Use the Custom operator

Take CPU backend as an example to use PyTorch Aten operator in Custom operator:

```python
# test_cpu_op.py
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

When using a PyTorch Aten `CPU` operator and `device_target` is `"GPU"`, should add prim attr like this:

```python
context.set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_cpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
op.add_prim_attr("primitive_target", "CPU")
```

> 1. Check compile tools exist and have right version when using cpp extension.
> 2. Make sure the build folder created by cpp extension is clean when using cpp extnsion first time.
> 3. Tested by PyTorch 1.9.1，cuda11.1，python3.7，download link:<https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl>, PyTorch cuda and local cuda should be same.