# 基于自定义算子接口调用第三方算子库

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/use_third_party_op.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API中的[Custom](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语方便快捷地进行不同类型自定义算子的定义和使用。

网络开发者可以根据需要选用不同的自定义算子开发方式。详情请参考Custom算子的[使用指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_custom.html)。

其中，自定义算子有一种开发方式`aot`方式有其特殊的使用方式。`aot`方式可以通过加载预编译的`so`来调用相应的`cpp`/`cuda`函数。因此，当第三方库提供了`cpp`/`cuda`函数`API`时，可以尝试将其函数接口在`so`中调用，以下以PyTorch的`Aten`库为例进行介绍。

## PyTorch Aten算子对接

当迁移一张使用PyTorch Aten算子的网络遇到内置算子不足的情况时，我们可以利用`Custom`算子的`aot`开发方式调用PyTorch Aten的算子进行快速验证。

PyTorch提供了一种方式可以支持引入PyTorch的头文件，从而使用其相关的数据结构编写`cpp/cuda`代码，并编译成`so`。参考：<https://pytorch.org/docs/stable/_modules/torch/utils/cpp_extension.html#CppExtension>。

将两种方式结合使用，自定义算子可以调用PyTorch Aten算子，使用方式如下:

### 1. 下载工程文件

工程文件可以通过[这里](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/migration_guide/test_custom_pytorch.tar)下载。

使用以下命令解压压缩包，得到文件夹`test_custom_pytorch`。

```bash
tar xvf test_custom_pytorch.tar
```

文件夹中包含以下几个文件：

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

使用PyTorch Aten算子主要关注env.sh、setup.py、leaky_relu.cpp/cu、test_*.py即可。

其中，env.sh用于设置环境变量，setup.py用于编译so，leaky_relu.cpp/cu用于参考编写调用PyTorch Aten算子的源码，test_*.py用于参考调用Custom算子。

### 2. 编写调用PyTorch Aten算子的源码文件

参考leaky_relu.cpp/cu，编写调用PyTorch Aten算子的源码文件。

由于`aot`类型的自定义算子采用`AOT`编译方式，要求网络开发者基于特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后在网络运行时框架会自动调用执行动态链接库中的函数。在算子实现的开发语言方面，`GPU`平台支持`CUDA`，`CPU`平台支持`C`和`C++`。源码文件中的算子实现函数的接口规范如下：

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);

```

如果是调用`cpu`算子，以`leaky_relu.cpp`为例，该文件提供`AOT`需要的函数`LeakyRelu`，里面调用了PyTorch Aten的函数`torch::leaky_relu_out`：

```cpp
#include <string.h>
#include <torch/extension.h> // 头文件引用部分
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
    // 如果使用不带输出的版本，代码如下：
    // torch::Tensor output = torch::leaky_relu(at_input);
    // at_output.copy_(output);
  return 0;
}

```

如果是调用`gpu`算子，以`leaky_relu.cu`为例：

```cpp
#include <string.h>
#include <torch/extension.h> // 头文件引用部分
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

其中，PyTorch Aten提供了带输出的算子函数版本和不带输出的算子函数版本，带输出的算子函数有`_out`后缀，PyTorch Aten提供了300+常用算子的`api`。

当调用`torch::*_out`时，不需要`output`拷贝。当调用不带`_out`后缀的版本，需要调用API`torch.Tensor.copy_`进行结果拷贝。

想查看支持调用PyTorch Aten的哪些函数，`CPU`版本参考PyTorch安装路径下的：`python*/site-packages/torch/include/ATen/CPUFunctions_inl.h` ，相应的`GPU`版本参考`python*/site-packages/torch/include/ATen/CUDAFunctions_inl.h`。

以上用例中使用了ms_ext.h提供的api，这里稍作介绍:

```cpp
// 将 MindSpore kernel 的 inputs/outputs 转换为 PyTorch Aten 的 Tensor
std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, c10::Device device) ;
```

### 3. 使用编译脚本`setup.py`生成so

setup.py使用PyTorch Aten提供的`cppextension`将上述`c++/cuda`源码编译成`so`文件。

执行前需要确保已经安装PyTorch。

```bash
pip install torch
```

并将PyTorch的`lib`加入`LD_LIBRARY_PATH`。

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
```

执行：

```bash
cpu: python setup.py leaky_relu.cpp leaky_relu_cpu.so
gpu: python setup.py leaky_relu.cu leaky_relu_gpu.so
```

将得到我们需要的 so 文件。

### 4. 使用自定义算子

以CPU为例，使用Custom算子调用上述PyTorch Aten算子，代码见test_cpu_op.py：

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

执行：

```bash
python test_cpu_op.py
```

结果：

```text
[[ 0.    -0.001]
 [-0.002  1.   ]]
```

注意：

若使用的是PyTorch Aten `GPU`算子，`device_target`需设置为`"GPU"`.

```python
context.set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_gpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
```

若使用的是PyTorch Aten `CPU`算子，而`device_target`是`"GPU"`，需要增加设置如下：

```python
context.set_context(device_target="GPU")
op = ops.Custom("./leaky_relu_cpu.so:LeakyRelu", out_shape=lambda x : x, out_dtype=lambda x : x, func_type="aot")
op.add_prim_attr("primitive_target", "CPU")
```

> 1. 使用cppextension编译so需满足该工具需要的编译器版本，检查gcc/clang/nvcc是否存在。
> 2. 使用cppextension编译so会在脚本路径生成一个build的文件夹，里面存放了so，脚本会将so拷贝到build外，但是cppextension如果发现build里已经有so会跳过编译，因此如果是新编译的so要记得清空build下的so。
> 3. 以上测试基于PyTorch 1.9.1版本，cuda使用11.1，python3.7，下载链接：<https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-linux_x86_64.whl>，PyTorch Aten支持的cuda版本需和本地的cuda版本一致，其他版本是否支持需用户自行探索。