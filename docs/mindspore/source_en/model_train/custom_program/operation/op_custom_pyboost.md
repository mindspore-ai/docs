# Custom Operators for Dynamic Graph Scenarios

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/custom_program/operation/op_custom_pyboost.md)

## Overview

In dynamic graph mode, network workflows are easier to debug, supporting operations like single-operator execution, normal functions/networks, and standalone gradient computations. For details about dynamic graphs, refer to [Dynamic Graph](https://www.mindspore.cn/docs/en/master/model_train/program_form/pynative.html).

While [Custom operator expressions](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/custom_program/operation/op_custom.ipynb) support both static and dynamic graphs, they require extensive definitions. MindSpore optimizes the custom operator definition for dynamic graphs to enhance usability and execution performance.

This guide demonstrates a multiplication operator implementation on Ascend platform. For related code and more examples, see [Repository Code](https://gitee.com/mindspore/mindspore/blob/master/tests/st/pynative/grad/test_custom_cpp_function_grad.py).

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

MindSpore provides foundational data structures and interfaces:

- `Function`: Computation function class template. Custom operator classes derive from this.
- `BaseTensor`: Tensor type. `BaseTensorPtr` and `BaseTensorPtrList` represent its pointer and list forms.
- `AutogradContext`: Autodiff environment, detailed below.
- `CustomLaunchAclnn`: Interface for invoking aclnn operators.

Include `ms_extension.h` and define classes within `mindspore::pynative` namespace to use these structures.

### Computation Function Class

The `Function` template facilitates forward and backward implementations. Users define:

```cpp
class CustomMul : public Function<CustomMul>
```

This class requires two methods: `Forward` (forward pass) and `Backward` (backward pass).

#### Forward Computation

The `Forward` method implements the operator's forward logic:

```cpp
static BaseTensorPtr Forward(AutogradContext *ctx, const BaseTensorPtr &x, const BaseTensorPtr &y)
```

1. **Computation**: Create an output tensor and invoke `aclnnMul` via `CustomLaunchAclnn`.
2. **Gradient Preparation**: Use `ctx->NeedGrad` to check if inputs require gradients. Save necessary tensors with `SaveForBackward`.

```cpp
auto output = std::make_shared<BaseTensor>(x->data_type(), BroadcastInferShape(x, y));
custom::CustomLaunchAclnn("aclnnMul", {x, y}, {output});

bool x_require_grad = ctx->NeedGrad(x);
bool y_require_grad = ctx->NeedGrad(y);
if (x_require_grad || y_require_grad) {
  ctx->SaveForBackward({x_require_grad ? y : nullptr, y_require_grad ? x : nullptr});
}
```

#### Backward Computation

The `Backward` method computes gradients:

```cpp
static BaseTensorPtrList Backward(AutogradContext *ctx, BaseTensorPtrList grad_outputs)
```

1. **Retrieve Saved Tensors**: Use `GetSavedTensors` to access tensors saved during forward pass.
2. **Gradient Calculation**: Compute gradients only for required inputs using `ctx->NeedsInputGrad`.

```cpp
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
```

### Computation Function and Python Binding

Define the computation function using `CustomMul::Apply`:

```cpp
BaseTensorPtr run_custom_mul(const tensor::BaseTensorPtr &x, const tensor::BaseTensorPtr &y) {
  return CustomMul::Apply(x, y);
}
```

Bind to Python via pybind11:

```cpp
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("mul", &mindspore::pynative::autograd::run_custom_mul, "Calculate the value x multiplied by y.");
}
```

## Operator Usage

Use `CustomOpBuilder` to compile and load the operator automatically:

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

**Output:**

```txt
out: 12.0
grads[0]: (Tensor(shape=[], dtype=Float32, value= 6), Tensor(shape=[], dtype=Float32, value= 4))
grads[1]: (Tensor(shape=[], dtype=Float32, value= 6),)
```

**Explanation:**

- `out`: Forward computation result.

- `grads[0]`: Gradients of inputs `x` and `y`.

- `grads[1]`: Gradient of parameter `p`.

This example demonstrates how to integrate custom operators into neural networks and compute gradients efficiently.