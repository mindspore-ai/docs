# 自定义算子（GPU）

`GPU` `模型开发`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/custom_operator_gpu.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

算子是构建神经网络的基本要素，当开发网络遇到内置算子无法满足要求时。你可以利用MindSpore方便地实现一个GPU算子。

- Primitive注册：算子原语是构建网络模型的基础单元，用户可以直接或者间接调用算子原语搭建一个神经网络模型。
- GPU Kernel实现：GPU Kernel用于调用GPU实现加速计算。
- GPU Kernel注册：算子注册用于将GPU Kernel及必要信息注册给框架，由框架完成对GPU Kernel的调用。

在本教程中，我们将在MindSpore框架中使用C++和CUDA开发一个TensorAddV2算子。TensorAddV2用于将两个同维度的Tensor逐元素相加。

## 注册算子原语

算子原语通常包括：

- 算子名：算子名用于唯一标识个算子
- 注释：描述算子的算法、使用约束。注释将被导出成为MindSpore API接口文档，供开发者查阅。
- 输入：算子输入Tensor。
- 属性：一般描述算法参数，例如Conv2d中`data_format`描述了输入数据为`NCHW`或者`NHWC`格式。
- 输入数据合法性校验：对输入数据、属性进行合法性校验，便于开发者及早发现网络模型存在的问题。
- 输出数据类型和维度推导：用于推导输出的数据类型和维度。

下面的代码中定义了一个名为TensorAddV2算子:

- `TensorAddV2`继承于`PrimitiveWithInfer`。
- `__init__`构造函数用于初始化算子，由于TensorAddV2没有属性，因此`__init__`没有额外输入。
- `infer_shape`方法中约束两个输入维度必须相同，输出的维度和x1的维度相同。
- `infer_dtype`方法中约束两个输入数据必须是float32类型，输出的数据类型和输入数据类型相同。

```python
# mindspore/ops/operations/math_ops.py
class TensorAddV2(PrimitiveWithInfer):
    """
    Adds two input tensors element-wise.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def infer_shape(self, x1_shape, x2_shape):
        validator.check_integer('input dims', len(x1_shape), len(x2_shape), Rel.EQ, self.name)
        for i in range(len(x1_shape)):
            validator.check_integer('input_shape', x1_shape[i], x2_shape[i], Rel.EQ, self.name)
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_type):
        validator.check_tensor_type_same({'x1_dtype': x1_dtype}, [mstype.float32], self.name)
        validator.check_tensor_type_same({'x2_dtype': x2_dtype}, [mstype.float32], self.name)
        return x1_dtype
```

接下来我们在__init__.py中导出TensorAddV2类型，方便用户在网络中导入使用。

```python
# mindspore/ops/operations/__init__.py
from .math_ops import (Abs, ACos, ..., TensorAddV2)
...
...
__all__ = [
  'ReverseSequence',
  'CropAndResize',
  ...,
  'TensorAddV2'
]
```

## GPU算子开发

GPU自定义算子继承于`GPUKernel`:

- `Init()`: 用于完成GPU Kernel的初始化，通常包括记录算子输入/输出维度，完成Launch前的准备工作。
- `GetInputSizeList()`: 向框架反馈输入Tensor需要占用的显存字节数。
- `GetOutputSizeList()`: 向框架反馈输出Tensor需要占用的显存字节数。
- `GetWorkspaceSizeList()`: 向框架反馈`Workspace`字节数，`Workspace`是用于计算过程中存放临时数据的空间。
- `Launch()`: 通常调用CUDA kernel(CUDA kernel是基于Nvidia GPU的并行计算架构开发的核函数)，或者cuDNN接口等方式，完成算子在GPU上加速。

下面的代码给出了TensorAddV2的实现：
为了支持数据类型的泛化，我们使用类模板定义`TensorAddV2GpuKernel`:

- `Init()`中记录了Tensor的元素个数。
- `GetInputSizeList()`返回了输入Tensor需要占用的字节数，TensorAddV2有两个Input，每个Input占用字节数为element_num * sizeof(T)。
- `GetOutputSizeList()`返回了输出Tensor需要占用的字节数，TensorAddV2有一个output，占用element_num * sizeof(T)字节。
- 由于TensorAddV2不需要`Workspace`，因此`GetWorkspaceSizeList()`返回空的`std::vector<size_t>`。
- `Launch()`接收input、output在显存的地址，接着调用`TensorAddV2`完成加速。

```c++
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

template <typename T>
class TensorAddV2GpuKernel : public GpuKernel {
 public:
  TensorAddV2GpuKernel() : element_num_(1) {}
  ~TensorAddV2GpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < shape.size(); i++) {
      element_num_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *x1 = GetDeviceAddress<T>(inputs, 0);
    T *x2 = GetDeviceAddress<T>(inputs, 1);
    T *y = GetDeviceAddress<T>(outputs, 0);

    TensorAddV2(element_num_, x1, x2, y, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    output_size_list_.push_back(element_num_ * sizeof(T));
  }

 private:
  size_t element_num_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
```

`TensorAddV2`中调用了CUDA kernel`TensorAddV2Kernel`来实现`element_num`个元素的并行相加:

```c++
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

 template <typename T>
 __global__ void TensorAddV2Kernel(const size_t element_num, const T* x1, const T* x2, T* y) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < element_num; i += blockDim.x * gridDim.x) {
    y[i] = x1[i] + x2[i];
  }
 }

 template <typename T>
 void TensorAddV2(const size_t &element_num, const T* x1, const T* x2, T* y, cudaStream_t stream){
    size_t thread_per_block = 256;
    size_t block_per_grid = (element_num + thread_per_block - 1 ) / thread_per_block;
    TensorAddV2Kernel<<<block_per_grid, thread_per_block, 0, stream>>>(element_num, x1, x2, y);
   return;
 }

 template void TensorAddV2(const size_t &element_num, const float* x1, const float* x2, float* y, cudaStream_t stream);
```

## GPU算子注册

算子信息包含:

- `Primive`
- `Input dtype, output dtype`
- `GPU Kernel class`
- `CUDA内置数据类型`

框架会根据`Primive`和`Input dtype, output dtype`，调用以`CUDA内置数据类型`实例化`GPU Kernel class`模板类。

如下代码中分别注册了支持float和int的TensorAddV2算子。

```c++
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.cc

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddOutputAttr(kNumberTypeFloat32),
                      TensorAddV2GpuKernel, float)

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddOutputAttr(kNumberTypeInt32),
                      TensorAddV2GpuKernel, int)

```

## 编译MindSpore

写好自定义GPU算子后，需要重新编译安装MindSpore，具体请参考[安装文档](https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_gpu_install_source.md#)。

## 算子验证

在教程的最后，我们构建一个单算子网络，来验证刚才开发的TensorAddV2算子:

```python
# tests/st/ops/gpu/test_tensoraddv2_op.py

import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as ops

context.set_context(device_target='GPU')

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_TensorAdd():
    x1 = Tensor(np.ones((3, 4), np.float32))
    x2 = Tensor(np.ones((3, 4), np.float32))
    y = ops.TensorAddV2()(x1, x2)
    print('result: ', y)
```

通过`pytest -s tests/st/ops/gpu/test_tensoraddv2_op.py::test_TensorAdd`命令执行后，可以看到结果符合预期:

```text
result: [[2. 2. 2. 2.]
  [2. 2. 2. 2.]
  [2. 2. 2. 2.]]
```

## 定义算子反向传播函数

如果算子要支持自动微分，需要在其原语中定义其反向传播函数（bprop）。你需要在bprop中描述利用正向输入、正向输出和输出梯度得到输入梯度的反向计算逻辑。反向计算逻辑可以使用内置算子或自定义反向算子构成。

定义算子反向传播函数时需注意以下几点：

- bprop函数的入参顺序约定为正向的输入、正向的输出、输出梯度。若算子为多输出算子，正向输出和输出梯度将以元组的形式提供。
- bprop函数的返回值形式约定为输入梯度组成的元组，元组中元素的顺序与正向输入参数顺序一致。即使只有一个输入梯度，返回值也要求是元组的形式。

例如，`TensorAddV2`的反向原语为：

```python
import mindspore.ops as ops
@bprop_getters.register(ops.TensorAddV2)
def get_bprop_tensoraddv2(self):
    """Generate bprop for TensorAddV2"""

    def bprop(x, y, out, dout):
        return dout, dout

    return bprop
```

在`test_tensoraddv2_op.py`文件中定义反向用例。

```python
import mindspore.ops as ops
class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=True)
        self.network = network

    def construct(self, x1, x2, sens):
        gout = self.grad(self.network)(x1, x2, sens)
        return gout

def test_grad_net():
    x1 = Tensor(np.ones((3, 4), np.float32))
    x2 = Tensor(np.ones((3, 4), np.float32))
    sens = Tensor(np.arange(3 * 4).reshape(3, 4).astype(np.float32))
    grad = Grad(Net())
    dx = grad(x1, x2, sense)
    print("dx[0]: ", dx[0].asnumpy())
```

执行用例:

```bash
pytest -s tests/st/ops/gpu/test_tensoraddv2_op.py::test_grad_net
```

执行结果:

```text
dx[0]: [[0. 1. 2. 3.]
        [4. 5. 6. 7.]
        [8. 9. 10. 11.]]
```
