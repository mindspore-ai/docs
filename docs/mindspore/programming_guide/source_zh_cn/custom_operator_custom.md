# 自定义算子（基于Custom表达）

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [自定义算子（基于Custom表达）](#自定义算子基于custom表达)
    - [概述](#概述)
    - [基本用法](#基本用法)
        - [akg类型的自定义算子开发](#akg类型的自定义算子开发)
        - [tbe类型的自定义算子开发](#tbe类型的自定义算子开发)
        - [aicpu类型的自定义算子开发](#aicpu类型的自定义算子开发)
        - [aot类型的自定义算子开发](#aot类型的自定义算子开发)
            - [GPU示例](#cpu示例)
            - [CPU示例](#gpu示例)
        - [pyfunc类型的自定义算子开发](#pyfunc类型的自定义算子开发)
    - [进阶用法](#进阶用法)
        - [算子信息注册](#算子信息注册)
        - [定义算子反向传播函数](#定义算子反向传播函数)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/custom_operator_custom.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API中的[Custom](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语方便快捷地进行不同类型自定义算子的定义和使用。

传统的添加一个自定义算子的方式，需要完成算子原语注册、算子实现、算子信息注册三部分工作。

其中：

- 算子原语：定义了算子在网络中的前端接口原型，也是组成网络模型的基础单元，主要包括算子的名称、属性（可选）、输入输出名称、输出shape推理方法、输出数据类型推理方法等信息。
- 算子实现：在Python侧定义函数（Ascend自定义算子）或C++侧定义类（GPU和CPU自定义算子），描述算子内部计算逻辑的实现。
- 算子信息：描述自定义算子的基本信息，如算子名称、支持的输入输出数据类型、支持的输入输出数据格式和属性等。它是后端做算子选择和映射时的依据。

相比于传统自定义算子方式，基于`Custom`原语自定义算子具有如下优势：

- 不同的自定义算子对应的算子原语都是`Custom`原语，无需对每个自定义算子定义一个相应的算子原语。上述提到的三部分工作可以在网络脚本中以统一的接口进行实现，并作为网络表达的一部分，不需要对MindSpore框架进行侵入式修改和重新编译。
- 实现了不同方式自定义算子的接口和使用统一，方便网络开发者根据需要灵活选用不同的自定义方式。
- 新增支持hybrid等自定义算子方式，并且可以跨平台使用。

## 基本用法

基于[Custom](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语的自定义算子支持的算子开发方式包括：akg、tbe、aot和pyfunc。

不同的算子开发方式差异如下：

| 算子开发方式 | 开发语言 | 编译方式 | 支持平台 | 推荐场景 |
| :------: | :------: |:------: | ------ | ------ |
| akg | MindSpore AKG DSL | JIT | `Ascend` `GPU` | Ascend/GPU平台普通场景 |
| tbe | TBE DSL | JIT | `Ascend` | Ascend平台场景 |
| aicpu | C/C++ | AOT | `Ascend` | Ascend平台场景 |
| aot | C/C++/CUDA | AOT | `GPU` `CPU` | GPU/CPU平台高性能场景 |
| pyfunc | Python | JIT | `CPU` | 快速算法验证、需要与Python进行交互等场景 |

> - DSL全称是Domain Specific Language。
> - AOT（Ahead Of Time）编译方式指的是，算子实现函数需提前被编译为动态链接库，然后在网络运行时由框架自动调用；JIT（Just In Time）编译方式则不需要提前编译算子实现函数，而是在网络编译或运行期间被框架直接调用。

不同的开发方式使用不同的开发语言实现算子计算逻辑，但是自定义算子的开发流程是一致的，包括算子实现、算子输出shape和数据类型推理和算子信息注册（可选）。网络开发者可以根据需要选用不同的自定义算子开发方式。下面分别介绍这几种自定义算子开发方式，每种开发方式均提供示例。

> 更多示例可参考MindSpore源码中[tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom)下的用例。

### akg类型的自定义算子开发

akg类型的自定义算子使用[MindSpore AKG](https://gitee.com/mindspore/akg)算子DSL，描述算子内部计算逻辑的实现。MindSpore AKG是基于TVM（Tensor Virtual Machine）和Polyhedral技术的算子开发和编译框架，支持Hybrid、IR builder和TVM compute等多种类型的算子DSL。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若算子包含属性或者只支持特定的输入输出数据类型或数据格式，则需要注册算子信息，算子信息生成方式请参考[算子信息注册](#算子信息注册)。若未注册算子信息，在后端做算子选择和映射的时候，将会从当前算子的输入中推导算子信息。

下面以test_custom_akg.py为例介绍akg类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_akg.py内容：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")

# 算子实现，Hybrid DSL
def add(a, b):
    c = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
    return c

if __name__ == "__main__":
    # 定义akg类型的自定义算子
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="akg")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- `context.set_context(device_target="GPU")`表示算子运行在GPU平台，若要运行在Ascend平台，请编译Ascend版本的MindSpore，并将device_target的值设置为"Ascend"。
- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```bash
python test_custom_akg.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

### tbe类型的自定义算子开发

tbe类型的自定义算子使用TBE（Tensor Boost Engine）算子DSL，描述算子内部计算逻辑的实现。算子DSL开发可以参考[TBE文档](https://support.huaweicloud.com/odevg-A800_3000_3010/atlaste_10_0063.html)。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

这种类型的自定义算子需要注册算子信息，算子信息生成方式请参考[算子信息注册](#算子信息注册)。

下面以test_custom_tbe.py为例介绍tbe类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_tbe.py内容：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp, custom_info_register

context.set_context(device_target="Ascend")

# 算子实现，注册算子信息
@custom_info_register(CustomRegOp() \
                      .input(0, "a") \
                      .input(1, "b") \
                      .output(0, "output") \
                      .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                      .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
                      .target("Ascend") \
                      .get_op_info())
def add(a, b, output, kernel_name="add"):
    import te.lang.cce
    from te import tvm
    data0 = tvm.placeholder(a.get("shape"), name="data0", dtype=a.get("dtype").lower())
    data1 = tvm.placeholder(b.get("shape"), name="data1", dtype=b.get("dtype").lower())
    res = te.lang.cce.vadd(data0, data1)
    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule(res)
    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data0, data1, res]}
    te.lang.cce.cce_build_code(sch, config)

if __name__ == "__main__":
    # 定义tbe类型的自定义算子
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="tbe")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 通过`CustomRegOp`生成算子信息，并通过`custom_info_register`装饰器注册算子信息。

执行用例：

```bash
python test_custom_tbe.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

### aicpu类型的自定义算子开发

aicpu类型的自定义算子采用AOT编译方式，要求算子开发者基于提供的特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后框架会根据开发者在算子属性中配置的动态链接库名称，找到对应动态链接库并加载算子。具体算子实现参考[CANN AICPU 自定义算子开发](https://support.huaweicloud.com/usermanual-mindstudio303/atlasms_02_0193.html)。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

这种类型的自定义算子需要注册算子信息，算子信息生成方式请参考[算子信息注册](#算子信息注册)，aicpu类型的自定义算子，需要额外指定`attr("cust_aicpu",  "required", "str", "mindspore_aicpu_kernels")`的属性，用于MindSpore找到对应的算子实现的动态链接库。

> - 需要注意的是，aicpu类型的自定义算子开发后编译成的动态链接库，需要存放到MindSpore的lib目录下，比如MindSpore安装在虚拟环境`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore`下，则aicpu的so文件需要放到`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore/lib/`目录下。
> - “cust_aicpu”的值为字符串，用算子动态链接库的名字去除`lib`前缀与`.so`后缀表示，如`libmindspore_aicpu_kernels.so`则设为`"mindspore_aicpu_kernels"`即可。

下面以test_dropout_aicpu.py为例介绍aicpu类型的自定义算子开发流程，其中自定义算子实现了dropout的功能，并且编译好的算子动态链接库，我们命名为libmindspore_aicpu_kernels.so，并已将该动态链接库放至mindspore根目录的lib下。

test_dropout_aicpu.py内容：

```python
import numpy as np
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import CustomRegOp, custom_info_register, DataType

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# 算子实现，注册算子信息
dropout2d_op_info = CustomRegOp("Dropout2D") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .output(1, "mask", "required") \
    .attr("keep_prob", "required", "float") \
    .attr("cust_aicpu", "required", "str", "mindspore_aicpu_kernels") \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.BOOL_Default) \
    .target("Ascend") \
    .get_op_info()

@custom_info_register(dropout2d_op_info)
def dropout2d_aicpu():
    """Dropout2D AiCPU register"""
    return

# 定义自定义算子网络
class NetDropout2D(nn.Cell):
    def __init__(self, keep_prob=0.5):
        super(NetDropout2D, self).__init__()
        self.op = ops.Custom(dropout2d_aicpu, out_shape=lambda x, _, cust_attr: (x, x), \
                              out_dtype=lambda x, _, cust_attr: (x, mstype.bool_), func_type="aicpu")
        self.keep_prob = keep_prob
        self.cust_aicpu_so_path = "mindspore_aicpu_kernels"

    def construct(self, inputs):
        return self.op(inputs, self.keep_prob,  self.cust_aicpu_so_path)

if __name__ == "__main__":
    # 定义aicpu类型的自定义算子
    input_tensor = Tensor(np.ones([1, 1, 2, 3]), mstype.float32)
    dropout2d_nn = NetDropout2D(0.5)
    output, mask = dropout2d_nn(input_tensor)
    print("output: ", output)
    print("mask: ", mask)
```

本例中，有如下几点需要说明：

- 可以用多种方式指定`Custom`原语的`out_shape`和`out_dtype`参数，可以给定类型，也可以用Python lambda函数等设置。本例中lambda函数表明输出的两个shape与输入相同，第一个输出的数据类型和输入张量的信息相同，第二个输出的数据类型为bool类型。
- 通过`CustomRegOp`生成算子信息，并通过`custom_info_register`装饰器注册算子信息。

执行用例：

```bash
python test_dropout_aicpu.py
```

执行结果：

```text
output : [[[[2.  2.  2.] [2.  2.  2.]]]]
mask: [[[[True  True  True]  [True  True  True]]]]
```

### aot类型的自定义算子开发

aot类型的自定义算子采用AOT编译方式，要求网络开发者基于特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后在网络运行时框架会自动调用执行动态链接库中的函数。在算子实现的开发语言方面，GPU平台支持CUDA，CPU平台支持C和C++。源码文件中的算子实现函数的接口规范如下：

```cpp
extern "C" int func_name(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

其中，函数名`func_name`可替换成任意有效函数名。返回值为int类型，约定0表示正常退出，非0表示发生异常。参数列表的含义如下：

- nparam (int): 输入输出总数。比如算子有2个输入，1个输出，则nparam的值为3。
- params (void \*\*): 输入输出指针数组。比如算子有2个输入，1个输出，params[0]指向第一个输入数据，params[1]指向第二个输入数据，params[2]指向输出数据。
- ndims (int \*): 输入输出shape维度数组。比如params[i]是个shape[1024, 1024]的张量，则ndims[i]的值为2。
- shapes (int64_t \*\*): 输入输出shape数组。比如params[i]是个shape[1024, 1024]的张量，则shapes[i][0]的值为1024，shapes[i][1]的值为1024。
- dtypes (const char \*\*): 输入输出数据类型数组。dtypes里的元素取值可为："float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "bool"。
- stream (void \*): CUDA流指针，仅定义GPU算子实现时需要。
- extra (void \*): 用于后续扩展。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](#算子信息注册)。

下面通过例子介绍GPU平台和CPU平台上aot类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

#### GPU示例

使用CUDA语言，编写算子实现的源码文件add.cu：

```cpp
#define THREADS 1024
__global__ void CustomAddKernel(float *input1, float *input2, float *output, size_t size) {
  auto idx = blockIdx.x * THREADS + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] + input2[idx];
  }
}

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 3) return 1;
  void *input1 = params[0];
  void *input2 = params[1];
  void *output = params[2];
  size_t size = 1;

  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  CustomAddKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(input1), static_cast<float *>(input2),
                                                   static_cast<float *>(output), size);
  return 0;
}
```

将add.cu编译成动态库add.so：

```bash
nvcc --shared -Xcompiler -fPIC -o add.so add.cu
```

编写测试用例test_custom_aot.py：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")

if __name__ == "__main__":
    # 定义aot类型的自定义算子
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 本例中需要将test_custom_aot.py和add.so放置在同一目录下，若add.so在其他目录，则需要将`Custom`第一个参数里路径修改为add.so的绝对路径。
- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```bash
python test_custom_aot.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

#### CPU示例

使用C或者C++语言，编写算子实现的源码文件add.cc：

```cpp
#include <string.h>
using size_t = decltype(sizeof(int));
using int64_t = decltype(sizeof(long));

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  if (nparam != 3) return 1;
  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  size_t size = 1;
  for (int i = 0; i < nparam; i++) {
    size *= shapes[2][i];
  }
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  for (int i = 0; i < size; i++) {
    output[i] = input1[i] + input2[i];
  }
  return 0;
}
```

将add.cc编译成动态库add.so：

```bash
g++ --shared -fPIC -o add.so add.cc
```

编写测试用例test_custom_aot.py：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="CPU")

if __name__ == "__main__":
    # 定义aot类型的自定义算子
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 本例中需要将test_custom_aot.py和add.so放置在同一目录下，若add.so在其他目录，则需要将`Custom`第一个参数里路径修改为add.so的绝对路径。
- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```bash
python test_custom_aot.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

### pyfunc类型的自定义算子开发

pyfunc类型的自定义算子使用原生Python语法定义算子实现函数，描述算子内部计算逻辑的实现。网络运行时框架会自动调用此函数。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](#算子信息注册)。

下面以test_custom_pyfunc.py为例介绍pyfunc类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_pyfunc.py内容：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="CPU")

def add(a, b):
    return a + b

if __name__ == "__main__":
    # 定义pyfunc类型的自定义算子
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="pyfunc")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```bash
python test_custom_pyfunc.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

## 进阶用法

### 算子信息注册

算子信息主要描述了算子实现函数所支持的输入输出类型、输入输出数据格式、属性和target（平台信息），它是后端做算子选择和映射时的依据。它通过[CustomRegOp](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop)接口定义，通过[custom_info_register](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.custom_info_register.html#mindspore-ops-custom-info-register)装饰器或者[Custom](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语构造函数中的`reg_info`参数，实现算子信息与算子实现函数的绑定，并最终注册到MindSpore C++侧的算子信息库。`reg_info`参数优先级高于`custom_info_register`装饰器。

算子信息中的target的值可以为"Ascend"或"GPU"或"CPU"，描述的是算子实现函数在当前target上所支持的输入输出类型、输入输出数据格式和属性等信息，对于同一个算子实现函数，其在不同target上支持的输入输出类型可能不一致，所以通过target进行区分。算子信息在同一target下只会被注册一次。

> - 算子信息中定义输入输出信息的个数和顺序、算子实现函数中的输入输出信息的个数和顺序，两者要完全一致。
> - 对于akg类型的自定义算子，若算子存在属性输入，则必须注册算子信息，算子信息中的属性名称与算子实现函数中使用的属性名称要一致；对于tbe类型的自定义算子，当前必须注册算子信息；对于aot类型的自定义算子，由于算子实现函数需要预先编译成动态库，所以无法通过装饰器方式绑定算子信息，只能通过`reg_info`参数传入算子信息。
> - 若自定义算子只支持特定的输入输出数据类型或数据格式，则需要注册算子信息，以便在后端做算子选择时进行数据类型和数据格式的检查。对于不提供算子信息的情况，则在后端做算子选择和映射的时候，将会从当前算子的输入中推导信息。

### 定义算子反向传播函数

如果算子要支持自动微分，需要定义其反向传播函数（bprop），然后将bprop函数传入`Custom`原语构造函数的`bprop`参数。你需要在bprop中描述利用正向输入、正向输出和输出梯度得到输入梯度的反向计算逻辑。反向计算逻辑可以使用内置算子或自定义Custom算子。

定义算子反向传播函数时需注意以下几点：

- bprop函数的入参顺序约定为正向的输入、正向的输出、输出梯度。若算子为多输出算子，正向输出和输出梯度将以元组的形式提供。
- bprop函数的返回值形式约定为输入梯度组成的元组，元组中元素的顺序与正向输入参数顺序一致。即使只有一个输入梯度，返回值也要求是元组的形式。

下面test_grad.py为例，展示反向传播函数的用法：

```python
import numpy as np
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 自定义算子正向实现
def square(x):
    y = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        y[i0] = y[i0] * y[i0]
    return y

# 自定义算子反向实现
def square_grad(x, dout):
    dx = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        dx[i0] = 2.0 * x[i0]
    for i0 in range(x.shape[0]):
        dx[i0] = dx[i0] * dout[i0]
    return dx

# 反向传播函数
def bprop():
    op = ops.Custom(square_grad, lambda x, _: x, lambda x, _: x, func_type="akg")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        # 定义akg类型的自定义算子，并提供反向传播函数
        self.op = ops.Custom(square, lambda x: x, lambda x: x, bprop=bprop(), func_type="akg")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    dx = ops.GradOperation(sens_param=True)(Net())(Tensor(x), Tensor(sens))
    print(dx)
```

其中：

- 反向传播函数中使用是的akg类型的自定义算子，算子定义与使用需要分开，即自定义算子在`custom_bprop`函数外面定义，在`custom_bprop`函数内部使用。

执行用例：

```bash
python test_grad.py
```

执行结果：

```text
[ 2.  8. 18.]
```

> 更多示例可参考MindSpore源码中[tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/master/tests/st/ops/graph_kernel/custom)下的用例。
