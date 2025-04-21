# Custom Operators for Dynamic Graph Scenarios

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/custom_program/operation/op_custom_pyboost.md)

## Overview

In dynamic graph mode, network workflows are easier to debug, supporting operations like single-operator execution, normal functions/networks, and standalone gradient computations.

While [Custom operator expressions](https://www.mindspore.cn/tutorials/en/br_base/custom_program/op_custom.html) support both static and dynamic graphs, they require extensive definitions. MindSpore optimizes the custom operator definition for dynamic graphs to enhance usability and execution performance.

This guide demonstrates a multiplication operator implementation on Ascend platform. For related code and more examples, see [Repository Code](https://gitee.com/mindspore/mindspore/blob/br_base/tests/st/pynative/grad/test_custom_cpp_function_grad.py).

## Operator Definition

To define a dynamic graph custom operator, users need to implement a C++ computation function and map it to Python via pybind11. Below is an example of a custom operator's computation function.

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

The `Function` class template constructs the computation function class `CustomMul`, which uses the `Apply` method. The C++ function `run_custom_mul` is bound to Python via `PYBIND11_MODULE` to create the custom operator.

### Data Structures and Interfaces

To facilitate user-defined operators, MindSpore provides foundational data structures and interfaces:

- `Function`: Computation function class template. Custom operator classes derive from this.
- `BaseTensor`: Tensor type. `BaseTensorPtr` and `BaseTensorPtrList` represent its pointer and list forms.
- `AutogradContext`: Autodiff environment, detailed below.
- `CustomLaunchAclnn`: Interface for invoking aclnn operators.

It should be noted that in order to use the data structures provided by MindSpore, it is necessary to refer to the header file `ms_extension.h` in the custom operator code and define the computational function class and computational function in the namespace `mindspore::pyboost`.

### Computation Function Class

In order to facilitate the implementation of user-defined operators and inverses, MindSpore provides a computational function class template `Function`. Users can define the following computational function class according to the operator class name they choose:

```c++
class CustomMul : public Function<CustomMul>
```

This class requires two methods: `Forward` (forward pass) and `Backward` (backward pass).

#### Forward Computation

The user implements the forward computation of a customized operator through the `Forward` method. First focus on the following function prototype. Its first input is fixed to `AutogradContext *`, and the rest of the inputs support `BaseTensorPtr`, `std::string`, or other base types, the number of which is determined by the number of inputs to the operator.

```c++
static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y)
```

Here is the forward function calculation part. The user first creates a Tensor with a data type of `x->data_type()` and a size of `BroadcastInferShape(x, y)`, then uses `CustomLaunchAclnn` to invoke the `aclnnMul` operator for computation. For knowledge related to the compilation of aclnn operators, you can refer to the relevant sections in [AOT type custom operators (Ascend platform)](https://www.mindspore.cn/tutorials/en/br_base/custom_program/operation/op_custom_ascendc.html#offline-compilation-and-deployment).

```c++
auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});
```

Finally save the forward inputs for the inverse function on which the differentiation algorithm depends. The `AutogradContext` class will be used here. First the `NeedGrad` interface is used to determine if the corresponding input needs to be derived. If there are inputs that need to be computed backward, record the information via `SaveForBackward`. For multiplication here, if `x` needs to be derived, `y` needs to be saved in the environment, and vice versa.

```c++
bool x_require_grad = ctx->NeedGrad(x);
bool y_require_grad = ctx->NeedGrad(y);
if (x_require_grad || y_require_grad) {
  ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
}
```

#### Backward Computation

The user implements the inverse computation of a customized operator through the `Backward` method. First focus on the following function prototype. Its first input is fixed to `AutogradContext *` and its second input is fixed to `BaseTensorPtrList`.

```c++
static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs)
```

First obtain the tensor used for the calculation of the inverse function, which comes from two parts: the list of tensors saved by the environment and the input of the inverse.
The tensor values saved by the environment are obtained by the `AutogradContext::GetSavedTensors` interface and correspond to the list of tensors recorded in the forward function using the `SaveForBackward` interface. Here the list of tensors recorded by the forward function is `{x_require_grad ? y : nullptr, y_require_grad ? x : nullptr}`, so `saved` has two elements.
The input to the inverse is the gradient of the forward input, which corresponds one-to-one to the output of the forward function. Here the forward function has only one output, so `dout` has only one element.

```c++
auto saved = ctx->GetSavedTensors();
auto dout = grad_outputs[0];
```

The value of each forward gradient is then calculated. To minimize the amount of computation, `ctx->NeedsInputGrad(i)` is used first to determine if the ith input needs to be derived. Only if it is needed does it go to the specific calculation function. The computation is done in the same way as the forward function computation can be done by calling the aclnn operator.

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

### Computation Function and Python Binding

After creating the computational function class `CustomMul` and its `Forward/Backward` methods, implement the computational function `run_custom_mul` for the custom operator. The `Apply` method of the `CustomMul` class needs to be used here, and its inputs need to correspond one-to-one with all inputs in the `CustomMul::Forward` signature except `AutogradContext`.

```c++
BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}
```

The C++ function `run_custom_mul` is linked to the Python function `mul` via `PYBIND11_MODULE`. Here, the inputs to `m.def` are respectively:

- `'mul'`: Corresponds to the Python function name.
- `&mindspore::pynative::autograd::run_custom_mul`: Corresponds to a C++ function pointer.
- `"Calculate the value x multiplied by y."`: Python function documentation.

```python
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "Calculate the value x multiplied by y.");
}
```

## Operator Usage

In order to facilitate the use of custom operators, MindSpore provides a Python class `CustomOpBuilder` to help users to implement automatic compilation and custom operator running and other functions. An example of how to use a custom operator is as follows.

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

Here, the user defines a custom operator module `self.my_ops = CustomOpBuilder(“my_ops”, ['. /custom_src/function_ops.cpp'], backend=“Ascend”).load()`. Here the meaning of `CustomOpBuilder` parameters are:

- `"my_ops"`: Customize the operator module name.
- `['./custom_src/function_ops.cpp']`: Customize the path to the operator C++ file. If there is more than one C++ file, you need to list them all in the list.
- `backend="Ascend"`: Customize the backend on which the operator runs.

It should be noted that the users need to call the `load` method after defining a custom operator using `CustomOpBuilder` for automatic compilation and loading of the operator.

Here the custom operator is called in the script via `self.my_ops.mul(x, y)`, where `mul` is the name of the Python function defined in `PYBIND11_MODULE` above.

Run the above script to get the results:

```txt
out: 12.0
grads[0]: (Tensor(shape=[], dtype=Float32, value= 6), Tensor(shape=[], dtype=Float32, value= 4))
grads[1]: (Tensor(shape=[], dtype=Float32, value= 6),)
```

In the above result, `out` denotes the positive output, the two `Tensors` of `grads[0]` denote the derivatives of the inputs `x` and `y`, respectively, and one `Tensor` of grads[1] denotes the derivative of the Parameter `p`.