# Custom Operators (CPU)

Translator: [JuLyAi](https://gitee.com/julyai)

`CPU` `Model Development`

<!-- TOC -->

- [Custom Operators (CPU)](#custom-operators-cpu)
    - [Overview](#overview)
    - [Registration Operator's Primitives](#registration-operators-primitives)
    - [Implementing CPU Operators and Registration Operators Information](#implementing-cpu-operators-and-registration-operators-information)
        - [Implementing CPU Operators](#implementing-cpu-operators)
        - [Registration Operators Information](#registration-operators-information)
    - [Editing MindSpore](#editing-mindspore)
    - [Using Custom CPU Operators](#using-custom-cpu-operators)
    - [Defining Operators' BProp Functions](#defining-operators-bprop-functions)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/custom_operator_cpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

When the built-in operators are not enough for developing the network, you can extend your custom CPU operators fast and conveniently using MindSpore's Python API and C++ API.

To add a custom operator, you need to complete 3 parts of the work, including operator primitives registration, operators implementation and operators information registration.

Among them:

- Operator primitives: Defining the front-end interface prototype of operators in the network; The basic unit of a network model, mainly including operator's name, attributes (optional), input / output name, output shape reasoning method, output dtype reasoning method, etc.
- Operators implementation: Using the C++ API provided by the framework and combining with the specific characteristics of the operators, the internal calculation logic of the operator can be realized.

This paper will take the custom `Transpose` operator as an example to introduce the steps of customizing operators.

## Registration Operator's Primitives

Each operator's primitive is a subclass inherited from the class `PrimitiveWithCheck`, whose type name is the operator's name.

The CPU operator primitives are defined under the path `mindspore/ops/operations`, and the appropriate file is selected according to the operator type. Definition of CPU operators' primitives' interface is as follows:

- Attributes are defined by the input parameters of construction function `__init__`. Operators in this use case have no init attributes, thus `__init__` has no additional input parameters.
- The input and output names are defined by the function `init_prim_io_names`.
- Checking shape of the output tensor is defined in `check_shape` function. Checking dtype of the output tensor is defined in `check_dtype` function.
- `_checkparam` file defines a series of operations for validity checking, such as value checking, type checking, etc.

Taking `Transpose` operator's primitive as an example, the following example codes are given.

```python
from mindspore.ops import PrimitiveWithInfer

class Transpose(PrimitiveWithInfer):
    """
    The definition of the Transpose primitive.
    """
    @prim_attr_register
    def __init__(self):
        """Initialize Transpose"""
        self.init_prim_io_names(inputs=['x', 'perm'], outputs=['output'])

    def infer_shape(self, x, perm):
        x_shape = x['shape']
        p_value = perm['value']
        if len(x_shape) != len(p_value):
            raise ValueError('The dimension of x and perm must be equal.')
        out_shapes = []
        for i in p_value:
            out_shapes.append(x_shape[i])
        return out_shapes

    def infer_dtype(self, x_dtype, perm_dtype):
        return x_dtype
```

## Implementing CPU Operators and Registration Operators Information

### Implementing CPU Operators

Usually, to implement a CPU operator needs to write a head file and a source file. The file path is `mindspore/ccsrc/backend/kernel_compiler/cpu`. If the logical realization of the operator is by calling the third-party library `MKL-DNN`, it will be placed in the subdirectory `mkldnn`. Please refer to [oneMkl](https://github.com/oneapi-src/oneMKL) and [oneDNN](https://github.com/oneapi-src/oneDNN) for details.

The head file of the operator contains the registration information of the operator and the declaration of the class. The operator class inherits from the parent class of `CPUKernel` and overloads `InitKernel` and `Launch`.

The source file of the operator is the implementation of the class. It mainly overloads the InitKernel and Launch functions. The head file example codes of the `Transpose` operator are as follows:

```cpp
class TransposeCPUFwdKernel : public CPUKernel {
 public:
  TransposeCPUFwdKernel() = default;
  ~TransposeCPUFwdKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> shape_;
  std::vector<int> axis_;
};
```

- The input parameters of the function `InitKernel` contain a constant reference to the node pointer. Through the member function of the class `AnfRuntimeAlgorithm`, the input and output shape of the operator node and the attribute information of the operator can be obtained.
- The input parameters of the function `Launch` are 3 vectors, including all the input addresses, workspace addresses and all the output addresses, respectively. The concrete implementation logic of the operator is described in the function body.
- `shape_` and `axis_` are 2 member variables defined.

The definition of the function `InitKernel` in the source file is as follows:

```cpp
void TransposeCPUFwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  axis_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "perm");
  if (shape_.size() != axis_.size()) {
    MS_LOG(EXCEPTION) << "The size of input shape and transpose axis shape must be equal.";
  }
}
```

- The functions in the class `AnfRuntimeAlgorithm` implement various operations on operator nodes. `shape_` represents the shape of the first input of the operator. `axis_` represents the attribute "perm" of the operator.
- The parameter "perm" of the`Transpose` operator's primitive is as an input, but "perm" is actually considered as the attribute of the operation when parsing.

> For details of the class `AnfRuntimeAlgorithm`, please refer to the declaration in MindSpore source codes under [mindspore/ccsrc/backend/session/anf_runtime_algorithm.h](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/ccsrc/backend/session/anf_runtime_algorithm.h).

The definition of the function `Launch` in the source file is as follows: First, get the address of each input and output in turn, and then transform the dimension according to `axis_`, and assign the value to the space pointed to by the output address.

```cpp
bool TransposeCPUFwdKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /*workspace*/,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  size_t size = IntToSize(inputs[0]->size / sizeof(float));
  size_t shape_size = IntToSize(shape_.size());
  if (shape_size > kMaxDim) {
    MS_LOG(EXCEPTION) << "Input is " << shape_size << "-D, but transpose supports max " << kMaxDim << "-D inputs.";
  }
  size_t pos_array[kMaxDim];
  size_t size_offset[kMaxDim];
  size_offset[0] = size / shape_[0];
  for (size_t i = 1; i < shape_size; i++) {
    size_offset[i] = size_offset[SizeToInt(i) - 1] / shape_[i];
  }
  for (size_t position = 0; position < size; position += 1) {
    size_t temp_position = position;
    pos_array[0] = temp_position / size_offset[0];
    for (size_t i = 1; i < shape_size; i++) {
      temp_position -= pos_array[SizeToInt(i) - 1] * size_offset[i - 1];
      pos_array[i] = temp_position / size_offset[i];
    }
    size_t new_position = pos_array[axis_[SizeToInt(shape_size) - 1]];
    size_t new_position_size = 1;
    for (int j = shape_size - 2; j >= 0; j--) {
      new_position_size *= shape_[axis_[j + 1]];
      new_position += pos_array[axis_[j]] * new_position_size;
    }
    output[new_position] = input[position];
  }
  return true;
}
```

### Registration Operators Information

Operators information is the key information to guide the back-end selection of implementing operators. The first parameter of `MS_REG_CPU_KERNEL` is the name of the registration operator, which is consistent with the operator name in the primitives. The second parameter indicates the type of each input and output in turn. The last parameter is the name of the class which the operators implement. `Transpose` operator registration codes are as follows:

```cpp
MS_REG_CPU_KERNEL(Transpose, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  TransposeCPUFwdKernel);
```

> The number and order of the input and output information defined in operator information, the number and order of input and output information in operator implementation, and the number and order of input and output name list in operator primitives should be consistent.

## Editing MindSpore

After writing the custom CPU operators, you need to recompile and reinstall MindSpore. For details, please refer to [Installation Document](https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_cpu_install_source.md#).

## Using Custom CPU Operators

After compiling and installing, the custom CPU operators can be used directly through the import primitives. Take the single operator network test of `Transpose` as an example.

Define the network in document `test_transpose.py`.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.transpose = ops.Transpose()

    def construct(self, data):
        return self.transpose(data, (1, 0))

def test_net():
    x = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    transpose = Net()
    output = transpose(Tensor(x))
    print("output: ", output)
```

Running case:

```bash
pytest -s test_transpose.py::test_net
```

Running results:

```text
output: [[0, 3]
        [1, 4]
        [2, 5]]
```

## Defining Operators' BProp Functions

If an operator needs to support automatic differentiation, its back-propagation function (bprop) needs to be defined in its primitives. You need to describe the reverse computing logic that uses forward input, forward output, and output gradient to get the input gradient in bprop. Reverse computation logic can be composed of built-in operators or custom reverse operators.

The following points should be paid attention to when defining operators' bprop functions:

- The order of input parameters of bprop function is defined as positive input, positive output and output gradient. If the operator is a multi-output operator, the forward output and output gradient will be provided in the form of tuples.
- The form of the return values of bprop function is arranged as a tuple composed of input gradient, and the order of elements in the tuple is consistent with that of forward input parameters. Even if there is only one input gradient, the return value must be in the form of tuples.

For example, the bprop primitives of `Transpose` are:

```python
import mindspore.ops as ops
invert_permutation = ops.InvertPermutation()
transpose = ops.Transpose()
zeros_like = ops.ZerosLike()
@bprop_getters.register(ops.Transpose)
def get_bprop_transpose(self):
    """Generate bprop for Transpose"""

    def bprop(x, perm, out, dout):
        return transpose(dout, invert_permutation(perm)), zeros_like(perm)

    return bprop
```

- `Transpose` bprop operator uses `InvertPermutation` operator, which also needs a complete process of primitives, registration and implementation like `Transpose` operator.

Define the bprop case in document `test_transpose.py`.

```python
import mindspore.ops as ops
class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout

def test_grad_net():
    x = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    sens = np.arange(2 * 3).reshape(3, 2).astype(np.float32)
    grad = Grad(Net())
    dx = grad(Tensor(x), Tensor(sens))
    print("dx: ", dx.asnumpy())
```

Running case:

```bash
pytest -s test_transpose.py::test_grad_net
```

Running results:

```text
dx:  [[0. 2. 4.]
     [1. 3. 5.]]
```
