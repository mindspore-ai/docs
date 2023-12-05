# aot类型自定义算子进阶用法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_zh_cn/operation/op_custom_aot.md)

## 概述

aot类型的自定义算子采用预编译的方式，要求网络开发者基于特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后在网络运行时框架会自动调用执行动态链接库中的函数。aot类型的自定义算子支持GPU平台的CUDA语言，和CPU平台的C和C++语言。关于aot类型的自定义算子开发的基础知识请参考[基础教程](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/operation/op_custom.html#aot类型的自定义算子开发)。

本教程中，我们将展示aot类型自定义算子的进阶功能，包括

- aot类型自定义算子的自编译功能；
- aot类型自定义算子的属性和中间变量；
- aot类型自定义算子的动态shape支持。

对于下面用例的完整代码，请查阅[这里](https://gitee.com/mindspore/mindspore/blob/r2.2/tests/st/ops/graph_kernel/custom/test_custom_aot_fused.py)。

## aot类型自定义算子进阶用法特性简介

### aot类型自定义算子的自动编译

当用户的aot类型自定义算子文件为单一文件，且编译时不需要自定义的编译选项时，可以使用自动编译功能。如此，用户可以给自定义算子提供算子实现的源文件，MindSpore会自动把源文件编译成二进制库进行调用。当前该功能支持基于GCC的C++文件编译和基于NVCC的CUDA文件编译。在使用自动编译功能的时候，有如下几点需要说明：

- MindSpore识别自动编译的方式为文件名后缀。为了使用自动编译功能，请使用后缀为`cpp`, `cc`或者`cu`的源文件。其他情况MindSpore将处理为二进制库的路径。
- 自动编译的结果在文件夹akg_kernel_meta下。
- 默认编译选项为：
    - C++: `g++ -std=c++17 --shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
    - CUDA 10: `nvcc --shared -Xcompiler -fPIC -O3 -gencode arch=compute_70, code=sm_70 --use_fast_math --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
    - CUDA 11（或者更高版本）: `nvcc --shared -Xcompiler -fPIC -O3 -gencode arch=compute_80, code=sm_80 --use_fast_math --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -I./ -o $object_path, $source_path`
- 由于MindSpore需要使用`-D_GLIBCXX_USE_CXX11_ABI=0`的编译选项，GPU平台下请避免使用版本低于10.1.168的CUDA软件栈。

### aot类型自定义算子的属性和中间变量

常用的算子当中，不少算子带有属性，比如convlution的kernel size、padding和strides。带有不同属性值的算子有着相同的计算逻辑，唯一的区别是初始化时赋予属性不同的数值。此外，在算子的计算过程中，可能需要一些额外的内存空间储存中间变量。下面的计算为例，如果我们考虑`input_1`和`input_2`计算`output`如下公式：

```python
tmp = Add(input_1, input_2)
output = ReduceSum(tmp, axis, keep_dims)
```

这里，我们需要在算子中添加如下中间变量和属性以在计算函数中使用，包括

- `tmp`为中间变量，记录加法的中间结果；
- `axis`是类型为`int`的属性，`keep_dims`是类型为`bool`的属性。

aot类型的自定义算子提供属性功能，如此，我们可以通过一套源码定义一类自定义算子。这类有着相同的计算逻辑，而通过算子初始化的时候对属性赋值达到不同的计算效果。此外，为了让MindSpore统一管理内存的分配和释放，aot类型的自定义算子提供了接口，指定中间变量占内存的大小，由MindSpore申请内存供计算使用。

### aot类型自定义算子的动态shape支持

动态Shape，指的是算子输入或者输出的形状依赖于具体的运算，无法在编译期提前计算得出。具体来说分两种情况：算子输入的形状在编译期未知和算子输出的形状依赖具体输入的值。算子输入的形状在编译期未知的场景较为常见。任何算子，无论其计算逻辑如何，只要在支持动态shape输入的网络中使用，都需要支持这种场景。

当前自定义算子aot模式支持算子输入的形状在编译期未知的动态shape场景，通过定义c++版本的shape推导函数支持自定义算子该场景下的类型推导。

值得注意的是，目前自定义算子尚不支持算子输出的形状依赖具体输入的值的动态shape场景。

## aot类型自定义算子进阶用法接口简介

### 主函数

源码文件中，算子实现函数的主函数必须满足如下规范：

```cpp
extern "C" int FuncName(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra);
```

其中，函数名`FuncName`可替换成任意有效函数名。返回值为int类型，约定0表示正常退出，非0表示发生异常。参数列表的含义如下：

- nparam (int): 输入，输出和中间变量总数。比如算子有2个输入，1个输出，1个中间变量，则nparam的值为4。
- params (void \*\*): 输入，输出和中间变量指针数组。比如算子有2个输入，1个输出，1个中间变量，那么params[0]指向第一个输入数据，params[1]指向第二个输入数据的内存，params[2]指向输出数据的内存，params[3]指向中间变量的内存。
- ndims (int \*): 输入，输出和中间变量shape维度数组。比如params[i]是个shape[1024, 1024]的张量，则ndims[i]的值为2。
- shapes (int64_t \*\*): 输入，输出和中间变量shape数组。比如params[i]是个shape[1024, 1024]的张量，则shapes[i][0]的值为1024，shapes[i][1]的值为1024。
- dtypes (const char \*\*): 输入，输出和中间变量数据类型数组。dtypes里的元素取值可为："float32"、"float16"、"float"、"float64"、"int"、"int8"、"int16"、"int32"、"int64"、"uint"、"uint8"、"uint16"、"uint32"、"uint64"和"bool"。
- stream (void \*): CUDA流指针，仅定义GPU算子实现时需要。
- extra_void (void \*): 属性相关数据结构指针。

### 初始化函数

为了支持算子属性和中间变量，我们需要定义算子初始化函数。算子初始化函数定义必须满足如下规范：

```cpp
extern "C" int FuncNameInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra);
```

其中，函数名`FuncName`为算子主函数的名字。返回值为int类型，约定0表示正常退出，非0表示发生异常。参数列表的含义如下：

- ndims (int \*): 输入输出shape维度数组。
- shapes (int64_t \*\*): 输入输出shape数组。
- dtypes (const char \*\*): 输入输出数据类型数组。
- extra (AotExtra \*): 用于带属性的自定义算子扩展。其中`AotExtra`类型定义在MindSpore提供的头文件[custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/r2.2/tests/st/ops/graph_kernel/custom/aot_test_files/custom_aot_extra.h)。

### Shape推导函数

为了支持动态shape，aot类型的自定义算子中需要加入C++版本的shape推导函数。算子shape推导函数定义必须满足如下规范：

```cpp
extern "C" std::vector<int64_t> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)
```

其中，函数名`FuncName`为算子主函数的名字。返回值为`std::vector<int64_t>`类型，为输出的shape。参数列表的含义如下：

- ndims (int \*): 输入shape维度数组。
- shapes (int64_t \*\*): 输入shape数组。
- extra (AotExtra \*): 用于带属性的自定义算子扩展。其中`AotExtra`类型定义在MindSpore提供的头文件[custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/r2.2/tests/st/ops/graph_kernel/custom/aot_test_files/custom_aot_extra.h)。

### 算子属性注册（Python）

算子属性的在初始化时的赋值通过算子注册文件实现。对于每一个属性，我们为算子注册文件创建一个`attr`，设置属性名和属性的值。其注册方法为

```python
def attr(self, name=None, param_type=None, value_type=None, default_value=None, **kwargs)
```

其参数含义参见[CustomRegOp](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop)相关接口文档。其中，在aot类型自定义算子注册时，我们注册时需要注意一下四个参数：

- name: aot类型自定义算子的属性的名称；
- param_type: 属性的参数类型。对于aot类型自定义算子的属性，这个输入固定为”required“，即必选参数；
- value_type: 属性的数值类型。对于aot类型自定义算子的属性，这个输入可以为具体的数值类型，也可以是"all"，即不限定类型；
- 最后一个输入需要指定输入名为`value=`，输入的值为属性的值。

## aot类型自定义算子进阶用法用例

下面我们用一个Add和ReduceSum的融合算子用例来介绍aot类型自定义算子的进阶用法。该算子先把两个输入相加，在对某个轴计算求和操作，其基本计算逻辑如下：

```python
tmp = Add(input_1, input_2)
output = ReduceSum(tmp, axis, keep_dims)
```

这里，我们需要在算子中添加如下中间变量和属性以在计算函数中使用，包括

- `tmp`为中间变量，记录加法的中间结果；
- `axis`是类型为`int`的属性，`keep_dims`是类型为`bool`的属性。

### 算子实现文件（C++/CUDA）:kernel.cc

为了实现算子，我们创建源文件`kernel.cc`，包括以下一个算子属性类`add_reduce_kernel_attr`和三个函数：`CustomKernelInit`、`CustomKernelInferShape`和`CustomKernel`。

#### 算子属性类

首先我们定义一个数据结构贮存算子属性，该数据接口继承自`AotKernelData`。`AotKernelData`是自定义算子属性数据结构的统一基类，通过下载MindSpore提供的头文件[custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/r2.2/tests/st/ops/graph_kernel/custom/aot_test_files/custom_aot_extra.h)放在源文件同一目录下并在文件前`#include "custom_aot_extra.h"`便可以使用相关接口。

```c++
#include <vector>
#include "custom_aot_extra.h"
class add_reduce_kernel_attr : public AotKernelData {
 public:
  int64_t axis;
  bool keep_dim;
};
```

这里我们在属性类`add_kernel`定义了：

- `axis`：成员变量，类型为`int64_t`；
- `keep_dim`：成员变量，类型为`bool`；

#### 算子初始化函数

定义完算子属性类后，我们定义算子初始化函数。值得注意是，这里的初始化函数名`CustomKernelInit`对应，那么下面对应函数的前缀应该都为`CustomKernel`。

```c++

extern "C" int CustomKernelInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  size_t workspace_size = 1;
  for (size_t i = 0; i < ndims[0]; i++) {
    workspace_size *= shapes[0][i];
  }

  std::vector<size_t> workspace = {workspace_size * sizeof(float)};
  extra->SetWorkSpace(workspace);

  add_reduce_kernel_attr *kernel_data_ptr = new add_reduce_kernel_attr;
  kernel_data_ptr->axis = extra->Attr<int64_t>("axis");
  kernel_data_ptr->keep_dim = extra->Attr<bool>("keep_dim");
  extra->SetKernelData(kernel_data_ptr);
  return 0;
}
```

这里我们需要一个中间变量`workspace`记录加法的中间结果，操作方式如下：

1. 计算`workspace`需要的内存大小：这里`workspace`的shape和第一个输入一样，因此先用`workspace_size *= shapes[0][i]`计算出`workspace`中元素的个数，再用`workspace_size * sizeof(float)`计算内存大小（这里默认元素类型为float）;
2. 把所有中间变量的内存大小储存在一个`std::vector<size_t>`类型的对象内：`std::vector<size_t> workspace = {workspace_size * sizeof(float)};`。这里因为只有一个中间变量，该向量只有一个元素；
3. 通过`AotExtra *extra`的`SetWorkSpace`设置中间变量内存大小：`extra->SetWorkSpace(workspace)`。

另外我们需要获得两个属性`axis`和`keep_dim`的值，操作方式如下：

1. 创建一个`add_reduce_kernel_attr`对象指针：`add_reduce_kernel_attr *kernel_ptr = new add_reduce_kernel_attr`。
2. 从`extra`中获取对应属性的值贮存在`kernel_ptr`中的成员变量中：`kernel_data_ptr->axis = extra->Attr<int64_t>("axis"); kernel_data_ptr->keep_dim = extra->Attr<bool>("keep_dim");`。这里`reduce_axis`和`keep_dim`分别为`int`和`bool`类型，我们用`extra->Attr<T>(std::string name)`接口的对应模板获取该类型属性的值。
    - 这里`T`支持类型为：`bool`、`string`、`int64_t`、`float`、`std::vector<int64_t>`、`std::vector<float>`、`std::vector<std::vector<int64_t>>`和`std::vector<std::vector<float>>`。
3. 把`kernel_ptr`存在`extra`中供算子计算时使用：`extra->SetKernelData(kernel_ptr)`。

#### 算子Shape推导函数

为了定义动态shape场景，我们定义C++版本的算子Shape推导函数如下。值得注意是，这里的算子Shape推导函数名`CustomKernelInferShape`和上面的初始化函数名`CustomKernelInit`的前缀均为前缀`CustomKernel`。

```c++
#include <vector>
#include "custom_aot_extra.h"

extern "C" std::vector<int64_t> CustomKernelInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  const int64_t kDynRankSize = -2;

  if (shapes[0][0] == kDynRankSize) {
    return std::vector<int64_t>{shapes[0][0]};
  }
  int64_t axis = extra->Attr<int64_t>("axis");
  bool keep_dim = extra->Attr<bool>("keep_dim");
  if (keep_dim) {
    if (axis == 0) {
      return std::vector<int64_t>{1, shapes[0][1]};
    } else {
      return std::vector<int64_t>{shapes[0][0], 1};
    }
  } else {
    return std::vector<int64_t>{shapes[0][1 - axis]};
  }
}
```

在上面的例子中，我们要注意：

- 根据MindSpore的规范，动态shape输入分为dynamic shape和dynamic rank两种情况，对应的shape输入分别为：
    - dynamic shape：输入的某一维的大小未知，用-1表示。例如输入的shape为[1024, -1, 1024]，表示输入为一个三维张量，第一维和第三维长度为1024，第二维长度位置；
    - dynamic rank：输入的维度的个数位置，输入的shape固定为[-2, ]。
- 为了支持C++的shape推导函数，需要处理输入为dynamic shape和dynamic rank的场景。例如上面的例子，如果输入为dynamic rank，那么输出也是dynamic rank。因此我们判断输入为[-2, ]时，直接返回[-2, ]。
- 对于输出shape依赖属性的场景，可以通过`extra->Attr<T>(std::string name)`模板接口获取属性。

#### 算子计算函数（主函数）

算子计算函数的接口规范和不带属性的自定义算子一样。值得注意是，这里的算子主函数名`CustomKernel`需要和上面的初始化函数名`CustomKernelInit`及算子Shape推导函数名`CustomKernelInferShape`对应。主函数和上面两个函数一起组成源文件`kernel.cc`。

```c++
extern "C" int CustomKernel(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra_void) {
  constexpr int OUTPUT_INDEX = 2;

  float *input_1 = static_cast<float *>(params[0]);
  float *input_2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  float *tmp = static_cast<float *>(params[3]);

  // Add
  int in_size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    in_size *= shapes[OUTPUT_INDEX][i];
  }

  for (int i = 0; i < in_size; i++) {
    tmp[i] = input_1[i] + input_2[i];
  }

  // ReduceSum
  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<add_reduce_kernel_attr *>(extra->KernelData());
  bool keep_dim = kernel_ptr->keep_dim;
  int64_t axis = kernel_ptr->axis;
  int64_t input_dim_1 = shapes[0][1];
  int size;
  if (keep_dim) {
    size = shapes[1][0] * shapes[1][1];
  } else {
    size = shapes[1][0];
  }

  int ext = shapes[0][axis];
  for (int i = 0; i < size; i++) {
    output[i] = 0;
    for (int j = 0; j < ext; j++) {
      int idx = input_dim_1 * (i * axis + j * (1 - axis)) + i * (1 - axis) + j * axis;
      output[i] = output[i] + tmp[idx];
    }
  }
  return 0;
}
```

在计算Add时我们使用了算子的中间变量，操作如下：

1. 把`params`数组中的指针依次类型转化为`float *`。根据上面接口的介绍，数组中的元素依次为：两个输入的地址指针(`input_1`和`input_2`)，一个输出的地址指针(`output`)，以及一个中间变量的地址指针(`tmp`)；
2. 把两个输入相加的结果存在中间变量中：`tmp[i] = input_1[i] + input_2[i]`。

在计算ReduceSum时我们使用了算子的属性值，操作如下：

1. 把`extra_void`类型转化为`AotExtra`类型指针：`AotExtra *extra = static_cast<AotExtra *>(extra_void)`。
2. 从`extra`中获取初始化函数中创立的`kernel_ptr`对象指针：`auto kernel_ptr = static_cast<add_reduce_kernel_attr *>(extra->KernelData())`。这里`extra->KernelData()`获得的是一个void对象指针，需要再进一步类型转化为`kernel_ptr`对象指针。
3. 使用`kernel_ptr`中储存的属性值进行计算：`bool keep_dim = kernel_ptr->keep_dim; int64_t axis = kernel_ptr->axis;`。这里我们从`kernel_ptr`获得变量`keep_dim`和`axis`进行计算。

### 算子定义文件:test_custom_aot.py

为了在MindSpore中添加aot类型的自定义算子调用上面函数，我们创建文件`test_custom_aot.py`。

```python
import numpy as np
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

class ReduceDynNet(Cell):
    def __init__(self, out_types, axis, keep_dim):
        super(ReduceDynNet, self).__init__()
        reduce_cpu_info = CustomRegOp("reduce_kernel_cpu") \
            .input(0, "x1") \
            .input(0, "x2") \
            .output(0, "y") \
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
            .attr("axis", "required", "all", value=axis) \
            .attr("keep_dim", "required", "all", value=keep_dim) \
            .target("CPU") \
            .get_op_info()
        # 由于上面定义了C++版本的shape推导函数，这里的ouptut_shape可以为`None`
        self.program = ops.Custom("./kernel.cc:CustomKernel", None, out_types, "aot", reg_info=reduce_cpu_info)

    def construct(self, x, y):
        return self.program(x, y)
```

该文件中的`ReduceDynNet`包括算子注册和算子定义两个部分。

#### 算子注册

算子属性的在初始化时的赋值通过算子注册文件实现。关于自定义算子注册的函数，参见[CustomRegOp](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop)相关文档。对于每一个属性，我们为算子注册文件`reduce_cpu_info`创建一个`attr`，设置属性名和属性的值。

这里每一个`attr`项有四个输入：第一个为名字，如`"axis"`或`"keep_dim"`；中间两个为`"required"`和`"all"`；最后一个输入需要指定输入名为`value=`，输入的值为属性的值，例如这里`value=axis`和`value=keep_dim`。这里我们从网络的输入确定这两个参数的值，这两个值应该和上面初始化函数和shape推导函数中使用的`extra->Attr<T>`模板接口的类型匹配。

此外，如果我们需要定义多个算子注册文件，需要使用不同的算子文件名，即`CustomRegOp`的入参，这里为`"add_with_attr_kernel_cpu"`。如果需要定义另一个算子原型相同但是属性值不同的算子时，该名字不能重复。

#### 算子定义

上面Python文件中通过自定义算子统一接口`Custom`定义了aot类型的自定义算子：`self.program = ops.Custom("./kernel.cc:CustomKernel", None, out_types, "aot", reg_info=reduce_cpu_info)`。因为我们前面定了C++版本的shape推导函数之后，这里的`ouptut_shape`可以为`None`.

值得注意的是，这里的算子定义中我们直接使用源文件名`./kernel.cc`，如此我们采用MindSpore提供的自动编译功能。注意这个时候要保证环境中存在对应的编译器（这里为g++，gpu环境的cu文件则需要nvcc）。

### 算子调用

作为测试，我们给`test_custom_aot.py`文件添加`__main__`函数如下：

```python
if __name__ == "__main__":
    shape = (4, 5)
    axis = 1
    keep_dim = False
    context.set_context(device_target="CPU")

    input_x = np.ones(shape).astype(np.float32)
    input_y = np.ones(shape).astype(np.float32)

    test = ReduceDynNet(mstype.float32, axis, keep_dim)
    dyn_x = Tensor(shape=[4, None], dtype=mstype.float32)
    # set the net to dynamic shape
    test.set_inputs(dyn_x, dyn_x)
    output = test(Tensor(input_x),Tensor(input_y))
    print(output)
```

执行文件调用算子：

```bash
python test_custom_aot.py
```

执行结果：

```text
[10. 10. 10. 10.]
```
