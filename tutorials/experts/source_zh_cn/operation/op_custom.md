# 自定义算子（基于Custom表达）

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_zh_cn/operation/op_custom.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## 概述

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API中的[Custom](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语方便快捷地进行不同类型自定义算子的定义和使用。

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

基于[Custom](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语的自定义算子支持的算子开发方式包括：hybrid、tbe、aicpu、aot、pyfunc、julia、akg。

不同的算子开发方式差异如下：

| 算子开发方式 | 开发语言              | 编译方式 | 支持平台 | 推荐场景                    |
|:-------|:------------------|:------ |:------ |:------------------------|
| [hybrid](#hybrid类型的自定义算子开发) | MindSpore HYBRID DSL | JIT | `Ascend` `GPU` `CPU` | 全平台通用开发和快速验证 |
| [tbe](#tbe类型的自定义算子开发)    | TBE DSL           | JIT | `Ascend` | Ascend AICORE自定义算子场景    |
| [aicpu](#aicpu类型的自定义算子开发)  | C/C++             | AOT | `Ascend` | Ascend AICPU自定义算子场景     |
| [aot](#aot类型的自定义算子开发)    | C/C++/CUDA        | AOT | `GPU` `CPU` | 高性能手写、对接调用第三方算子库场景      |
| [pyfunc](#pyfunc类型的自定义算子开发) | Python            | JIT | `CPU` | 快速算法验证、需要与Python进行交互等场景 |
| [julia](#julia类型的自定义算子开发)  | Julia             | JIT | `CPU` | 科学计算场景、需要使用Julia编程等场景   |
| [akg](#akg类型的自定义算子开发)    | MindSpore AKG DSL | JIT | `Ascend` `GPU` | 用于开发验证场景，不建议普通用户使用      |

> - DSL全称是Domain Specific Language。
> - AOT（Ahead Of Time）编译方式指的是，算子实现函数需提前被编译为动态链接库，然后在网络运行时由框架自动调用；JIT（Just In Time）编译方式则不需要提前编译算子实现函数，而是在网络编译或运行期间被框架直接调用。

不同平台的不同场景下的推荐开发方式如下：

- Ascend: hybrid（通用场景），aicpu（不规则运算的高性能实现）；
- GPU: hybrid（通用场景），aot（基于CUDA的高性能实现）；
- CPU: hybrid（通用场景），aot（基于C++的高性能实现）。

不同的开发方式使用不同的开发语言实现算子计算逻辑，但是自定义算子的开发流程是一致的，包括算子实现、算子输出shape和数据类型推理和算子信息注册（可选）。网络开发者可以根据需要选用不同的自定义算子开发方式。下面分别介绍这几种自定义算子开发方式，每种开发方式均提供示例。

> 更多示例可参考MindSpore源码中[tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/r2.0.0-alpha/tests/st/ops/graph_kernel/custom)下的用例。

### Hybrid类型的自定义算子开发

Hybrid类型的自定义算子是自定义算子的默认定义类型。通过使用Hybrid类型的自定义算子，用户可以用类Python的语法描述算子计算逻辑，且无需关注MindSpore框架对于算子定义的工程细节，让用户专注于算法本身。

Hybrid类型的自定义算子使用[MindSpore Hybrid DSL](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/ms_kernel.html#语法规则)描述算子内部计算逻辑的实现。用MindSpore Hybrid DSL定义的函数可以被[AKG算子编译器](https://gitee.com/mindspore/akg)解析进行JIT编译生成高效算子，在大规模模型的训练推理中使用。同时，用MindSpore Hybrid DSL定义的函数可以当做一个`numpy`函数直接调用，方便用户调试的同时也可以灵活的切换到[pyfunc 类型的自定义算子](#pyfunc类型的自定义算子开发)，做到一次开发，多个模式多个平台多个场景复用的自定义算子表达。

下面用例(test_custom_hybrid.py)介绍hybrid类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。
值得注意的是，Hybrid类型的自定义算子采取源码变换的方式打通MindSpore的图编译器和算子编译器，用户可以直接使用MindSpore Hybrid DSL提供的关键词，例如下面的`output_tensor`，而无需引入对应Python函数。更多MindSpore Hybrid DSL关键词的介绍，参见[MindSpore Hybrid DSL关键词](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/ms_kernel.html#关键词)。

```python
import numpy as np
from mindspore import ops
import mindspore as ms
from mindspore.ops import ms_kernel

ms.set_context(device_target="GPU")

# 算子实现，Hybrid DSL
@ms_kernel
def add(a, b):
    c = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
    return c

if __name__ == "__main__":
    # 定义hybrid类型的自定义算子(Custom的默认模式)
    op = ops.Custom(add)

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- Hybrid类型是Custom的默认类型。
- Hybrid类型自定义算子的输入必须是一个带有[`@ms_kernel`](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.kernel.html)的函数。
- Hybrid类型自定义算子定义时可以使用自带的自动shape/dtype推导函数，也可以手动输入shape/dtype推导函数。

执行用例：

```bash
python test_custom_hybrid.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

### tbe类型的自定义算子开发

tbe类型的自定义算子使用TBE（Tensor Boost Engine）算子DSL，描述算子内部计算逻辑的实现。算子DSL开发可以参考[TBE文档](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/operatordev/tbedevg/tbedevg_000003.html)。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

这种类型的自定义算子需要注册算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)。

下面以test_custom_tbe.py为例介绍tbe类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_tbe.py内容：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp, custom_info_register

ms.set_context(device_target="Ascend")

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
    output = op(ms.Tensor(x0), ms.Tensor(x1))
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

aicpu类型的自定义算子采用AOT编译方式，要求算子开发者基于提供的特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后框架会根据开发者在算子属性中配置的动态链接库名称，找到对应动态链接库并加载算子。具体算子实现参考[CANN AICPU 自定义算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/operatordev/aicpudevg/aicpudevg_000026.html)。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

这种类型的自定义算子需要注册算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)，aicpu类型的自定义算子，需要额外指定`attr("cust_aicpu",  "required", "str", "mindspore_aicpu_kernels")`的属性，用于MindSpore找到对应的算子实现的动态链接库。

> - 需要注意的是，aicpu类型的自定义算子开发后编译成的动态链接库，需要存放到MindSpore的lib目录下，比如MindSpore安装在虚拟环境`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore`下，则aicpu的so文件需要放到`/home/conda/envs/aicpu/lib/python3.7/site-packages/mindspore/lib/`目录下。
> - “cust_aicpu”的值为字符串，用算子动态链接库的名字去除`lib`前缀与`.so`后缀表示，如`libmindspore_aicpu_kernels.so`则设为`"mindspore_aicpu_kernels"`即可。

下面以test_dropout_aicpu.py为例介绍aicpu类型的自定义算子开发流程，其中自定义算子实现了dropout的功能，并且编译好的算子动态链接库，我们命名为libmindspore_aicpu_kernels.so，并已将该动态链接库放至mindspore根目录的lib下。

test_dropout_aicpu.py内容：

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import CustomRegOp, custom_info_register, DataType

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

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
                              out_dtype=lambda x, _, cust_attr: (x, ms.bool_), func_type="aicpu")
        self.keep_prob = keep_prob
        self.cust_aicpu_so_path = "mindspore_aicpu_kernels"

    def construct(self, inputs):
        return self.op(inputs, self.keep_prob,  self.cust_aicpu_so_path)

if __name__ == "__main__":
    # 定义aicpu类型的自定义算子
    input_tensor = ms.Tensor(np.ones([1, 1, 2, 3]), ms.float32)
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

执行结果（由于dropout算子具有随机性，多次运行结果存在差异）：

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

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)。

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
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="GPU")

if __name__ == "__main__":
    # 定义aot类型的自定义算子
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
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
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="CPU")

if __name__ == "__main__":
    # 定义aot类型的自定义算子
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
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

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)。

下面以test_custom_pyfunc.py为例介绍pyfunc类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_pyfunc.py内容：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="CPU")

def add(a, b):
    return a + b

if __name__ == "__main__":
    # 定义pyfunc类型的自定义算子
    op = ops.Custom(add, out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="pyfunc")

    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
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

### julia类型的自定义算子开发

julia类型的自定义算子使用Julia语法定义算子实现函数，描述算子内部计算逻辑的实现。网络运行时框架会自动调用执行相应的Julia函数。

算子输出shape和数据类型推导可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)。

下面以两个输入张量相加为例，介绍julia类型的自定义算子开发流程:

首先，用户需要通过单独文件实现Julia函数，如(add.jl)：

```julia
# add.jl
module Add
# inputs: x, y, output: z, output should use .= to inplace assign
function add(x, y, z)
    z .= x + y
end
end
```

其次，在网络脚本中通过自定义算子方式引用上面所写的Julia函数，以test_custom_julia.py为例：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="CPU")

if __name__ == "__main__":
    # 定义julia类型的自定义算子
    op = ops.Custom("./add.jl:Add:add", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="julia")
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型和第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```bash
python test_custom_julia.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

注意事项：

1. 用户需确保下载正确版本的Julia，即version>=1.6.0。
2. 由于运行时调用的Julia C api是从`libjulia.so`中获取的，因此需要用户设置`julia/lib`到`LD_LIBRARY_PATH`，以julia-1.6.5为例:

   ```bash
   # download julia-1.6.5
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.5-linux-x86_64.tar.gz
   # extract file
   tar xvf julia-1.6.5-linux-x86_64.tar.gz
   # if $JULIA_DIR not exist
   export LD_LIBRARY_PATH=$PWD/julia-1.6.5/lib:$LD_LIBRARY_PATH
   # else
   export LD_LIBRARY_PATH=$JULIA_DIR/lib:$LD_LIBRARY_PATH
   ```

3. `Custom` 第一个入参指定用户书写的Julia函数需按照`file_name:module_name:func_name`格式指定，`file_name`需包含文件路径，建议使用绝对路径。
4. Julia代码文件需包含`module`, `module`内包含`function`，且`module`/`function`都以`end`结束。
5. Julia函数的输入输出顺序需与算子的输入输出顺序一致。
6. Julia函数的最终输出，即kernel output的赋值需要使用`.=`，否则结果无法写入内存。
7. Julia代码支持[Julia](https://docs.julialang.org/en/v1/)的常用语法，用户需自行保证语法正确，函数可正确执行。
8. 用户想在Julia文件内使用Julia的第三方软件包，需自行下载对应软件以确保能正确调用，可以通过 `import pkg; pkg.add("somepkg")`进行安装。
9. `julia array`在内存上是`column major`排列的，而`numpy array`是`row major`排列的，如果Julia和numpy做比较，非elemwise计算需考虑内存排布。在Julia函数中，可以通过如下代码示例进行`numpy array`和`julia array`的相互转换:

   ```julia
   function change_input_to_row_major(x)
       return permutedims(reshape(x, reverse(size(x))), length(size(x)):-1:1)
   end

   function change_output_to_row_major(x)
       return reshape(permutedims(x, length(size(x)):-1:1), size(x))
   end
   ```

   以矩阵乘为例：

   ```julia
   # julia array is column-major, numpy array is row-major
   # user should change julia or numpy's layout to keep same behavior
   #= EXAMPLE
   A[2,3]               B[3,4]               C[2,4]
   NUMPY:
   [[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
    [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                     [9,10,11,12]]
   JULIA:
   change_input_to_row_major:
   1.inputs read numpy data from memory:
   [[1, 3, 5]       [[1, 4, 7,10]
    [2, 4, 6]]       [2, 5, 8,11]
                     [3, 6, 9,12]]
   2.inputs after reshape(reverse(shape)):
   [[1, 4]          [[1, 5, 9]
    [2, 5]           [2, 6,10]
    [3, 6]]          [3, 7,11]
                     [4, 8,12]]
   3.inputs after transpose/permutedims:
   [[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
    [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                     [9,10,11,12]]
   change_output_to_row_major:
   1.output after transpose/permutedims:
                                          [[38, 83]
                                           [44, 98]
                                           [50,113]
                                           [56,128]
   2.output after reshape:
                                          [[38, 50, 83, 113]
                                           [44, 56, 98, 128]]
   3.output read numpy data from memory:
                                          [[38, 44, 50,  56]
                                           [83, 98,113, 128]]
   =#
   function foo!(x, y, z)
       x = change_input_to_row_major(x)
       y = change_input_to_row_major(y)
       z .= gemm(x, y, z)
       z .= change_output_to_row_major(z)
   end
   ```

### akg类型的自定义算子开发

akg类型的自定义算子使用[MindSpore AKG](https://gitee.com/mindspore/akg)算子DSL，描述算子内部计算逻辑的实现。MindSpore AKG是基于TVM（Tensor Virtual Machine）和Polyhedral技术的算子开发和编译框架，支持Hybrid、IR builder和TVM compute等多种类型的算子DSL。

算子输出shape和数据类型推理可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若算子包含属性或者只支持特定的输入输出数据类型或数据格式，则需要注册算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/operation/op_custom_adv.html#算子信息注册)。若未注册算子信息，在后端做算子选择和映射的时候，将会从当前算子的输入中推导算子信息。

下面以test_custom_akg.py为例介绍akg类型的自定义算子开发流程，其中自定义算子实现两个输入张量相加的功能。

test_custom_akg.py内容：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="GPU")

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
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- `set_context(device_target="GPU")`表示算子运行在GPU平台，若要运行在Ascend平台，请编译Ascend版本的MindSpore，并将device_target的值设置为"Ascend"。
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

## 开发用例

开发前，我们导入 MindSpore 的相关依赖。

```python
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import ms_kernel
from mindspore.nn import Cell

#############################################
# 这里选用你使用的平台类型：CPU, GPU 或者 Ascend #
#############################################

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
# 选用CPU使用下面一行
# ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
# 选用Ascend使用下面一行
# ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
```

### 用例一：基于pyfunc模式定义sin算子

首先，我们写一个基于numpy的计算正弦函数的Python原生函数。

```python
def sin_by_numpy(x):
    return np.sin(x)
```

然后我们要定义两个函数，一个是张量形状的推导函数（infer_shape），另一个是张量数据类型的推导函数（infer_dtype）。这里要注意：

- 张量形状的推导函数是输入张量的形状；
- 张量数据类型的推导函数是输入张量的数据类型。

```python
def infer_shape(x):

    #    1. 这里的输入x是算子输入张量的形状
    #    2. sin函数是逐元素计算，输入的形状和输出的一样
    return x

def infer_dtype(x):

    #    1. 这里的输入x是算子输入张量的数据类型
    #    2. sin函数输入的数据类型和输出的一样
    return x

```

下面我们用上面的函数自定义一个算子，其输入包括

- func：自定义算子的函数表达，这里我们用`sin_by_numpy`函数；
- out_shape: 输出形状的推导函数，这里我们用`infer_shape`函数；
- out_dtype: 输出数据类型的推导函数，这里我们用`infer_dtype`函数；
- func_type: 自定义算子类型，这里我们用`"pyfunc"`。

然后我们调用算子计算结果。

```python
sin_by_numpy_op = ops.Custom(func = sin_by_numpy,     # 这里填入自定义算子的函数表达
                             out_shape = infer_shape, # 这里填入输出形状的推导函数
                             out_dtype = infer_dtype, # 这里填入输出数据类型的推导函数
                             func_type = "pyfunc"     # 这里填入自定义算子类型
                            )
input_tensor = ms.Tensor([0,1, 0.2, 0.3, 0.4], dtype=ms.float32)
result_cus = sin_by_numpy_op(input_tensor)
print(result_cus)
```

我们可以得到结果为，即上面输入对应的sin值。

```text
[0.         0.84147096 0.19866933 0.29552022 0.38941833]
```

### 用例二：利用hybrid类型的自定义算子实现三维张量的加法函数

首先，我们写一个基于MindSpore Hybrid DSL书写一个计算三维张量相加的函数。

注意：

- 对于输出张量使用 `output_tensor`，用法为：`output_tensor(shape, dtype)`；
- 所有的计算需要基于标量计算，如果是Tensor对象，那么需要写清楚所有index；
- 基本循环的写法和Python一样，循环维度的表达可以使用 `range`。

```python
@ms_kernel
def tensor_add_3d(x, y):
    result = output_tensor(x.shape, x.dtype)
    #    1. 你需要一个三层循环
    #    2. 第i层循环的上界可以用x.shape[i]获得
    #    3. 你需要基于每个元素表达计算，例如加法为 x[i, j, k] + y[i, j, k]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                result[i, j, k] = x[i, j, k] + y[i, j, k]

    return result
```

下面我们用上面的函数自定义一个算子。

注意到基于`ms_kernel`的`hybrid`函数时，我们可以使用自动的形状和数据类型推导。

因此我们只用给一个`func`输入（`func_type`的默认值为`"hybrid"`）。

```python
tensor_add_3d_op = ops.Custom(func = tensor_add_3d)
input_tensor_x = ms.Tensor(np.ones([2, 3, 4]).astype(np.float32))
input_tensor_y = ms.Tensor(np.ones([2, 3, 4]).astype(np.float32) * 2)
result_cus = tensor_add_3d_op(input_tensor_x, input_tensor_y)
print(result_cus)
```

同时我们可以使用`pyfunc`模式验证上面定义的正确性。

这里我们不需要重新定义算子计算函数`tensor_add_3d`，直接将`func_type`改为`"pyfunc"`即可。

注意`pyfunc`模式时我们需要手写类型推导函数。

```python
def infer_shape_py(x, y):
    return x

def infer_dtype_py(x, y):
    return x

tensor_add_3d_py_func = ops.Custom(func = tensor_add_3d,
                                   out_shape = infer_shape_py,
                                   out_dtype = infer_dtype_py,
                                   func_type = "pyfunc")

result_pyfunc = tensor_add_3d_py_func(input_tensor_x, input_tensor_y)
print(result_pyfunc)
```

我们可以得到如下结果，即两个Tensor的和。

```text
 [[[3. 3. 3. 3.]
  [3. 3. 3. 3.]
  [3. 3. 3. 3.]]

 [[3. 3. 3. 3.]
  [3. 3. 3. 3.]
  [3. 3. 3. 3.]]]
```
