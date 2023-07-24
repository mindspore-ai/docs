# 自定义算子（CPU）

`Linux` `CPU` `模型开发` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/custom_operator_cpu.md)

## 概述

当开发网络遇到内置算子不足以满足需求时，你可以利用MindSpore的Python API和C++ API方便快捷地扩展CPU端的自定义算子。

添加一个自定义算子，需要完成算子原语注册、算子实现、算子信息注册三部分工作。

其中：

- 算子原语：定义了算子在网络中的前端接口原型，也是组成网络模型的基础单元，主要包括算子的名称、属性（可选）、输入输出名称、输出shape推理方法、输出dtype推理方法等信息。
- 算子实现：利用框架提供的C++ API，结合算子具体特性实现算子内部计算逻辑。
- 算子信息：描述CPU算子的基本信息，如算子名称、支持的输入输出类型等。它是后端做算子选择和映射时的依据。

本文将以自定义`Transpose`算子为例，介绍自定义算子的步骤。

## 注册算子原语

算子的原语是一个继承于`PrimitiveWithInfer`的子类，其类型名称即是算子名称。

CPU算子原语定义在`mindspore/ops/operations`路径下，根据算子类型选择适合的文件，接口定义如下：

- 属性由构造函数`__init__`的入参定义。本用例的算子没有init属性，因此`__init__`没有额外的入参。
- 输入输出的名称通过`init_prim_io_names`函数定义。
- 输出Tensor的shape和dtype检验在`__infer__`函数中实现。

以`Transpose`算子原语为例，给出如下示例代码。

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
    def __infer__(self, x, perm):
        x_shape = x['shape']
        p_value = perm['value']
        if len(x_shape) != len(p_value):
            raise ValueError('The dimension of x and perm must be equal.')
        out_shapes = []
        for i in p_value:
            out_shapes.append(x_shape[i])
        out = {'shape': tuple(out_shapes),
               'dtype': x['dtype'],
               'value': None}
        return out
```

## 实现CPU算子和注册算子信息

### 实现CPU算子

通常一个CPU算子的实现，需要编写一个头文件和一个源文件，文件路径为`mindspore/ccsrc/backend/kernel_compiler/cpu`，如果算子的逻辑实现是通过调用第三方库`MKL-DNN`，则放在子目录`mkldnn`下。详细介绍请参考[oneMKL](https://github.com/oneapi-src/oneMKL)和[oneDNN](https://github.com/oneapi-src/oneDNN) 。

算子的头文件中包括算子的注册信息和类的声明。算子类继承于`CPUKernel`父类，重载`InitKernel`和`Launch`两个成员函数。

算子的源文件是类的实现，主要是重载InitKernel和Launch两个函数，`Transpose`算子实现的头文件代码示例如下：

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

- `InitKernel`函数的入参包含一个节点指针的常量引用，通过`AnfRuntimeAlgorithm`类的成员函数可以获取该算子节点输入输出的shape和算子的属性信息等。
- `Launch`函数的入参是三个向量，分别包含所有的输入地址，workspace地址，所有的输出地址。函数体中描述算子的具体实现逻辑。
- `shape_`和`axis_`是定义的两个成员变量。

源文件中`InitKernel`函数的定义如下：

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

- `AnfRuntimeAlgorithm`类中的函数实现了各种对算子节点的操作，`shape_`表示算子第1个输入的shape，`axis_`表示算子的属性perm。
- `Transpose`算子原语中参数“perm”作为输入传入，但是在解析时元组类型的“perm”实际被认为是算子的属性。

> `AnfRuntimeAlgorithm`类的详细内容可参考MindSpore源码中[mindspore/ccsrc/backend/session/anf_runtime_algorithm.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/backend/session/anf_runtime_algorithm.h)下的声明。

源文件中`Launch`函数的定义如下：首先依次获取每个输入输出的地址，然后根据`axis_`变换维度，把值赋给输出地址指向的空间。

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

### 注册算子信息

算子信息是指导后端选择算子实现的关键信息，`MS_REG_CPU_KERNEL`中第一个参数是注册算子的名称，和原语中算子名称一致，第二个参数依次指明每个输入输出的类型，最后一个参数是算子实现的类名。`Transpose`算子注册代码如下：

```cpp
MS_REG_CPU_KERNEL(Transpose, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  TransposeCPUFwdKernel);
```

> 算子信息中定义输入输出信息的个数和顺序、算子实现中的输入输出信息的个数和顺序、算子原语中输入输出名称列表的个数和顺序，三者要完全一致。

## 编译MindSpore

写好自定义CPU算子后，需要重新编译安装MindSpore，具体请参考[安装文档](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_cpu_install_source.md#)。

## 使用自定义CPU算子

编译并安装完成后，自定义CPU算子可以通过导入原语直接使用。下面以`Transpose`的单算子网络测试为例进行说明。

在`test_transpose.py`文件中定义网络。

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

执行用例:

```bash
pytest -s test_transpose.py::test_net
```

执行结果:

```text
output: [[0, 3]
        [1, 4]
        [2, 5]]
```

## 定义算子反向传播函数

如果算子要支持自动微分，需要在其原语中定义其反向传播函数（bprop）。你需要在bprop中描述利用正向输入、正向输出和输出梯度得到输入梯度的反向计算逻辑。反向计算逻辑可以使用内置算子或自定义反向算子构成。

定义算子反向传播函数时需注意以下几点：

- bprop函数的入参顺序约定为正向的输入、正向的输出、输出梯度。若算子为多输出算子，正向输出和输出梯度将以元组的形式提供。
- bprop函数的返回值形式约定为输入梯度组成的元组，元组中元素的顺序与正向输入参数顺序一致。即使只有一个输入梯度，返回值也要求是元组的形式。

例如，`Transpose`的反向原语为：

```python
import mindspore.ops as ops
invert_permutation = ops.InvertPermutation()
transpose = ops.Transpose()
zeros_like = ops.zeros_like()
@bprop_getters.register(ops.Transpose)
def get_bprop_transpose(self):
    """Generate bprop for Transpose"""

    def bprop(x, perm, out, dout):
        return transpose(dout, invert_permutation(perm)), zeros_like(perm)

    return bprop
```

- `Transpose`反向算子中用到了`InvertPermutation`算子，该算子和`Transpose`算子开发一样，需要有算子的原语，注册，实现等完整的流程。

在`test_transpose.py`文件中定义反向用例。

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

执行用例:

```bash
pytest -s test_transpose.py::test_grad_net
```

执行结果:

```text
dx:  [[0. 2. 4.]
     [1. 3. 5.]]
```
