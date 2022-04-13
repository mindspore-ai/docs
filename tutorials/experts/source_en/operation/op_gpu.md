# Custom Operators (GPU)

Translator: [Leon_02](https://gitee.com/Leon_02)

`GPU` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/op_gpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Operator is the basic element of constructing neural network. When built-in operators cannot meet requirements during network development, you can utilize MindSpore to quickly extend custom operators of the Graphics Processing Unit.

- Primitive registration: the register operator primitive is the basic unit of constructing network model. Users can directly or indirectly call the operator primitive to build a neural network model.
- GPU Kernel implementation: GPU kernel is used to call GPU to accelerate computing.
- GPU Kernel registration: operator registration is used to register the GPU kernel and necessary information to the framework, and the framework completes the call to the GPU kernel.

In this tutorial, we will develop a TensorAddV2 operator using C++ and CUDA in the mindspore framework. TensorAddV2 is used to add two tensors of the same dimension element by element.

## Registering the Operator Primitive

Operator primitives usually include:

- Aperator names: operator names are used to uniquely identify operators.
- Annotations: describe the algorithm and usage constraints of operators. The annotations will be exported as Mindspore API interface documentation for developers to refer to.
- Input: the tensor(s) for operator input.
- Attributes: for example, the `data_format` attribute in Conv2d describes that the input data is in `NCHW` or `NHWC` format.
- Validation of input data: verify the validity of input data and attributes, which is convenient for developers to find the problems of network model as soon as possible.
- Output data type and dimension derivation: used to derive the data type and dimension of output.

The following code defines an operator called TensorAddV2:

- `TensorAddV2` is a subclass inherited from `PrimitiveWithInfer`.
- The constructor `__init__` is used to initialize the operator, since TensorAddV2 doesn't have any attributes, there is none additional input for `__init__`.
- The function `infer_shape` constraints two input dimensions must be the same and the output dimension will be same as the dimension of x1.
- The function `infer_dtype` constrains that two input data must be of type float32 and the output data type is the same as the input data type.

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

Next we'll export TensorAddV2 type in '__init__.py', which convenient for users to import and use in the network.

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

## Implementing a GPU operator

Custom GPU operators inherit from `GPUKernel`:

- `Init()`: it is used to initialize the GPU kernel, usually includes recording the input / output dimension of the operator, and completing the preparation before launch.
- `GetInputSizeList()`: feedback to the frame the number of bytes of video memory to input tensor.
- `GetOutputSizeList()`: feedback to the frame the number of bytes of video memory to output tensor.
- `GetWorkspaceSizeList()`: feedback to the frame the number of bytes for `Workspace`, where `Workspace` is the space used to store temporary data during calculation.
- `Launch()`: generally, CUDA kernel (CUDA kernel is a kernel function developed by Nvidia GPU's parallel computing architecture) or cudnn interface are called to complete the operator acceleration on GPU.

The following code shows the implementation of TensorAddV2:
In order to support generalization of data types, we use class template to define `TensorAddV2GpuKernel`:

- `Init()` records the number of tensor elements.
- `GetInputSizeList()` returns the number of bytes the input tensor needs to occupy. TensorAddV2 has two Input and the number of bytes per input equals to element_num * sizeof(T).
- `GetOutputSizeList()` returns the number of bytes the output tensor needs to occupy. TensorAddV2 has one output and the output occupies element_num * sizeof(T) bytes.
- Since TensorAddV2 doesn't need `Workspace`, the `GetWorkspaceSizeList()` returns a null `std::vector<size_t>`.
- `Launch()` receives the addresses of input and output in video memory, and then calls `TensorAddV2` to complete acceleration.

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

`TensorAddV2` calls CUDA kernel`TensorAddV2Kernel` to implement the parallel addition of `element_num` elements:

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

## Registering the Operator Information

Operator information includes:

- `Primive`
- `Input dtype, output dtype`
- `GPU Kernel class`
- `CUDA built-in dtype`

Framework calls `CUDA built-in dtype` to instantiate `GPU Kernel class` template class based on `Primive` and `Input dtype, output dtype`.

The TensorAddV2 operators supporting float and int are registered in the code below:

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

## Compiling Mindspore

After writing the custom GPU operator, you need to recompile and install MindSpore, see [Installation Documentation](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_source_en.md#).

## Operator verification

At the end of the tutorial, we construct a single operator network to validate the TensorAddV2 operator we just developed：

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

When the command `pytest -s tests/st/ops/gpu/test_tensoraddv2_op.py::test_TensorAdd` executes, you can see the results meeting expectations：

```text
result: [[2. 2. 2. 2.]
  [2. 2. 2. 2.]
  [2. 2. 2. 2.]]
```

## Defining Operators' BProp Functions

If an operator needs to support automatic differentiation, its back-propagation function (bprop) needs to be defined in its primitives. You need to describe the reverse computing logic that uses forward input, forward output, and output gradient to get the input gradient in bprop. Reverse computation logic can be composed of built-in operators or custom reverse operators.

The following points should be paid attention to when defining operators' bprop functions:

- The order of input parameters of bprop function is defined as positive input, positive output and output gradient. If the operator is a multi-output operator, the forward output and output gradient will be provided in the form of tuples.
- The form of the return values of bprop function is arranged as a tuple composed of input gradient, and the order of elements in the tuple is consistent with that of forward input parameters. Even if there is only one input gradient, the return value must be in the form of tuples.

For example, the bprop primitives of `TensorAddV2` are:

```python
import mindspore.ops as ops
@bprop_getters.register(ops.TensorAddV2)
def get_bprop_tensoraddv2(self):
    """Generate bprop for TensorAddV2"""

    def bprop(x, y, out, dout):
        return dout, dout

    return bprop
```

Define the bprop case in document `test_tensoraddv2_op.py`.

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

Running case:

```bash
pytest -s tests/st/ops/gpu/test_tensoraddv2_op.py::test_grad_net
```

Running results:

```text
dx[0]: [[0. 1. 2. 3.]
        [4. 5. 6. 7.]
        [8. 9. 10. 11.]]
```
