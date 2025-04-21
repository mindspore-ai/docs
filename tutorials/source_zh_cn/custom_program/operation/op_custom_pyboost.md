# 动态图场景的自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/custom_program/operation/op_custom_pyboost.md)

## 概述

动态图模式下，网络流程更容易调试，可以支持执行单算子、普通函数和网络，以及单独求梯度等操作。

基于[Custom的自定义算子表达](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/custom_program/op_custom.html)虽然可以同时支持静态图和动态图，但是需要定义的内容较多。因此MindSpore针对动态图的自定义算子定义方式做了优化，方便用户使用的同时，还能提升自定义算子的执行性能。

下面以一个昇腾平台的乘法算子为例讲解，相关算子文件和更多用例参见[仓库代码](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/pynative/grad/test_custom_cpp_function_grad.py)。

## 算子定义

为了定义一个动态图的自定义算子，用户需要定义一个C++的计算函数，然后通过pybind11将C++计算映射到Python作为MindSpore算子使用。下面是一个自定义算子的计算函数样例。

```cpp
#include <string>
#include "ms_extension.h"

namespace mindspore::pynative {
namespace autograd {
ShapeVector BroadcastInferShape(const BaseTensorPtr &t1, const BaseTensorPtr &t2) {
  ShapeVector s1 = t1->shape();
  ShapeVector s2 = t2->shape();
  ShapeVector out_shape(std::max(s1.size(), s2.size()), 1LL);
  if (out_shape.empty()) {
    return out_shape;
  }
  for (size_t i = out_shape.size(); i > 0; i--) {
    if (i <= s1.size() && s1[s1.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s1[s1.size() - i];
    } else if (i <= s2.size() && s2[s2.size() - i] > 1) {
      out_shape[out_shape.size() - i] = s2[s2.size() - i];
    }
  }
  return out_shape;
}

class CustomMul : public Function<CustomMul> {
 public:
  static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y) {
    auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
    custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});
    bool x_require_grad = ctx->NeedGrad(x);
    bool y_require_grad = ctx->NeedGrad(y);
    if (x_require_grad || y_require_grad) {
      ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
    }
    return output;
  }

  static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs) {
    auto saved = ctx->GetSavedTensors();
    auto dout = grad_outputs[0];

    BaseTensorPtr grad_x = nullptr;
    BaseTensorPtr grad_y = nullptr;

    if (ctx->NeedsInputGrad(0)) {
      grad_x = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[0]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[0]}, {grad_x});
    }
    if (ctx->NeedsInputGrad(1)) {
      grad_y = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[1]));
      custom::CustomLaunchAclnn("aclnnMul", {dout, saved[1]}, {grad_y});
    }

    return {grad_x, grad_y};
  }
};

BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}

}  // namespace autograd
}  // namespace mindspore::pynative

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "Calculate the value x multiplied by y.");
}
```

这里使用计算函数类模板`Function`构建了一个计算函数类`CustomMul`，并使用计算函数类中的`Apply`方法定义计算函数，最后通过`PYBIND11_MODULE`将C++函数`run_custom_mul`链接到Python函数`mul`中构建自定义算子。

### 数据结构与接口

为了方便用户定义算子，MindSpore提供了基础的数据结构和接口，包括：

- `Function`：计算函数类模板。自定义算子的计算函数类均由此类派生出来。
- `BaseTensor`：张量。`BaseTensorPtr`为对应的指针的数据结构，`BaseTensorPtrList`为对应的指针的列表的数据结构。
- `AutogradContext`：自动微分环境。这个数据结构的用法将在下面详细介绍。
- `CustomLaunchAclnn`：调用aclnn算子接口。

值得注意的是，为了使用MindSpore提供的数据结构，需要在自定义算子代码里引用头文件`ms_extension.h`，并将计算函数类和计算函数定义在命名空间`mindspore::pyboost`中。

### 计算函数类

为了方便用户实现自定义算子及反向，MindSpore提供计算函数类模板`Function`。用户使用时，可根据自己选择的算子类名，定义如下计算函数类：

```c++
class CustomMul : public Function<CustomMul>
```

对于这个计算类，用户只需要定义两个方法，分别对应算子的正向计算与反向计算。

#### 正向计算

用户通过`Forward`方法实现自定义算子的正向计算。首先关注如下函数原型。其第一个输入固定为`AutogradContext *`，其余输入支持`BaseTensorPtr`、`std::string`，或者其它基础类型，其个数由算子的输入个数决定。

```c++
static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y)
```

下面是正向函数计算部分。用户先创建一个数据类型为`x->data_type()`，大小为`BroadcastInferShape(x, y)`的`Tensor`，然后使用`CustomLaunchAclnn`调用`aclnnMul`算子进行计算。对于aclnn算子的编译相关知识，可以参考[AOT类型自定义算子（Ascend平台）](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/custom_program/operation/op_custom_ascendc.html#编译与部署方法)中的相关章节。

```c++
auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});
```

最后为反向函数保存微分算法依赖的正向输入。这里会使用`AutogradContext`类。首先通过`NeedGrad`接口确定对应输入是否需要求导。如果有输入需要计算反向，则通过`SaveForBackward`记录相关信息。这里的乘法，如果`x`需要求导，则需在环境中保存`y`，反之亦然。

```c++
bool x_require_grad = ctx->NeedGrad(x);
bool y_require_grad = ctx->NeedGrad(y);
if (x_require_grad || y_require_grad) {
  ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
}
```

#### 反向计算

用户通过`Backward`方法实现自定义算子的反向计算。首先关注如下函数原型。其第一个输入固定为`AutogradContext *`，第二个输入固定为`BaseTensorPtrList`。

```c++
static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs)
```

首先获取反向函数计算使用的张量，张量的内容来自两个部分：环境保存的张量列表与反向的输入。
环境保存的张量值由`AutogradContext::GetSavedTensors`接口获得，对应正向函数中使用`SaveForBackward`接口记录的张量列表。这里正向函数记录的张量列表为`{x_require_grad ? y : nullptr, y_require_grad ? x : nullptr}`，因此`saved`有两个元素。
反向的输入为正向输入的梯度，与正向函数的输出一一对应。这里正向函数只有一个输出，因此`dout`只有一个元素。

```c++
auto saved = ctx->GetSavedTensors();
auto dout = grad_outputs[0];
```

然后计算每一个正向梯度的值。为了尽可能的减少计算量，先使用`ctx->NeedsInputGrad(i)`判断第i个输入是否需要求导。如果需要才会进入具体的计算函数。其计算方式与正向函数计算一样可以调用aclnn算子进行计算。

```c++
if (ctx->NeedsInputGrad(0)) {
  grad_x = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[0]));
  custom::CustomLaunchAclnn("aclnnMul", {dout, saved[0]}, {grad_x});
}
if (ctx->NeedsInputGrad(1)) {
  grad_y = std::make_shared<BaseTensor>(dout->data_type(), BroadcastInferShape(dout, saved[1]));
  custom::CustomLaunchAclnn("aclnnMul", {dout, saved[1]}, {grad_y});
}
```

### 计算函数及Python绑定

在创建完计算函数类`CustomMul`及其`Forward/Backward`方法后，实现自定义算子的计算函数`run_custom_mul`。这里需要使用`CustomMul`类的`Apply`方法，其输入需要与`CustomMul::Forward`签名中的除了`AutogradContext`之外的所有输入一一对应。

```c++
BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}
```

然后通过`PYBIND11_MODULE`将C++函数`run_custom_mul`链接到Python函数`mul`中。这里，`m.def`的输入分别为：

- `'mul'`：对应Python函数名字。
- `&mindspore::pynative::autograd::run_custom_mul`：对应C++函数指针。
- `"Calculate the value x multiplied by y."`：Python函数文档。

```python
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "Calculate the value x multiplied by y.");
}
```

## 算子使用

为了方便用户使用自定义算子，MindSpore提供了Python类`CustomOpBuilder`帮助用户实现自动编译及自定义算子运行等功能。一个自定义算子的使用用例如下。

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn
from mindspore.ops import CustomOpBuilder

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(2.0, requires_grad=True)
        self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

    def construct(self, x, y):
        z = self.my_ops.mul(x, y)
        return self.my_ops.mul(z, self.p)


x = Tensor(1.0, ms.float32) * 2
y = Tensor(1.0, ms.float32) * 3
net = MyNet()
grad_op = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params())
out, grads = grad_op(x, y)
print('out:', out)
print('grads[0]:', grads[0])
print('grads[1]:', grads[1])
```

这里，用户定义了一个自定义算子模块`self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()`。这里`CustomOpBuilder`的参数含义分别为：

- `"my_ops"`：自定义算子模块名。
- `['./custom_src/function_ops.cpp']`：自定义算子C++文件路径。如果有多个C++文件，需要在列表中一一列出。
- `backend="Ascend"`：自定义算子运行的后端。

值得注意的是，在使用`CustomOpBuilder`定义完自定义算子后需要调用`load`方法进行算子的自动编译和加载。

这里在脚本中通过`self.my_ops.mul(x, y)`调用自定义算子，其中`mul`为上面`PYBIND11_MODULE`中定义的Python函数名。

运行以上脚本，获得结果：

```txt
out: 12.0
grads[0]: (Tensor(shape=[], dtype=Float32, value= 6), Tensor(shape=[], dtype=Float32, value= 4))
grads[1]: (Tensor(shape=[], dtype=Float32, value= 6),)
```

上面结果中，`out`表示正向的输出，`grads[0]`的两个`Tensor`分别表示输入`x`和`y`的导数，grads[1]的一个`Tensor`表示Parameter `p`的导数。

