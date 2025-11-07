# Tensor View Mechanism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/view.md)

## Overview

A view operation creates a new tensor that shares the same data storage as the original tensor but has a different shape or layout. In other words, a view operation does not copy data but holds the existing data from a different perspective.

Core Features:

- **Memory Sharing**: The new tensor created by a view operation shares the underlying data storage with the original tensor.
- **Zero Copy**: No data is copied, avoiding memory allocation overhead.
- **Shape Transformation**: The shape of a tensor can be changed without altering its data content.

## Tensor View Methods

### Tensor.view()

The `Tensor.view()` method is used to quickly and efficiently change the shape of a Tensor.

```python
import mindspore
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ViewNet(nn.Cell):
    def construct(self, x):
        # Change shape using view
        y = x.view(-1)  # Flatten to a 1D tensor
        # Change to another shape
        z = x.view(3, 2)  # Change to a 3x2 tensor
        return y, z

# Create the original tensor
x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
net = ViewNet()
y, z = net(x)

print("Original tensor:", x)
print("Original shape:", x.shape)
print("After flattening:", y)
print("New shape:", y.shape)
print("Changed tensor:", z)
print("New shape:", z.shape)
```

The running result is as follows:

```text
Original tensor: [[1 2 3]
 [4 5 6]]
Original shape: (2, 3)
After flattening: [1 2 3 4 5 6]
New shape: (6,)
Changed tensor: [[1 2]
 [3 4]
 [5 6]]
New shape: (3, 2)
```

### Tensor.view_as()

The `Tensor.view_as()` method reshapes a tensor to have the same shape as another target tensor.

```python
import mindspore
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ViewAsNet(nn.Cell):
    def construct(self, x, target):
        # Change x to have the same shape as target
        return x.view_as(target)

x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
target = mindspore.tensor([[0, 0], [0, 0], [0, 0]])

net = ViewAsNet()
y = net(x, target)

print("Result:", y)
print("Target shape:", target.shape)
print("Result shape:", y.shape)
```

The running result is as follows:

```text
Result: [[1 2]
 [3 4]
 [5 6]]
Target shape: (3, 2)
Result shape: (3, 2)
```

## Notes

### Tensor Contiguity

The view operation requires the tensor to be contiguous in memory. If the input tensor is not contiguous, or if the output of `view` needs to be passed to an operator that does not support view inputs, the `contiguous()` method must be called first to ensure the tensor is contiguous.

```python
import mindspore

x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
y = mindspore.mint.transpose(x, 0, 1)  # Create a non-contiguous tensor

# Check for contiguity
print("Is y contiguous:", y.is_contiguous())

# Use contiguous() to ensure contiguity
z = y.contiguous().view(-1)
print("Is z contiguous:", z.is_contiguous())
```

The running result is as follows:

```text
Is y contiguous: False
Is z contiguous: True
```

### Consistent Total Number of Elements

The view operation requires the total number of elements in the new shape to be the same as in the original tensor.

```python
import mindspore

x = mindspore.tensor([1, 2, 3, 4, 5, 6])
print("Number of elements in original tensor:", x.numel())

# Correct: 6 = 2 * 3
y = x.view(2, 3)
print("Changed to 2x3:", y)

# Error: 6 ≠ 2 * 4
try:
    z = x.view(2, 4)
except ValueError as e:
    print("Shape mismatch error:", e)
```

The running result is as follows:

```text
Number of elements in original tensor: 6
Changed to 2x3: [[1 2 3]
 [4 5 6]]
Shape mismatch error: ValueError: The accumulate of x_shape must be equal to out_shape, but got x_shape: [const vector]{6}, and out_shape: [const vector]{2, 4}
```

## view vs. reshape

- **view operation**:
    - **Strict Contiguity Requirement**: The View operation requires that the tensor must be contiguous in memory.
    - **Failure Mechanism**: If the tensor is not contiguous, the view operation will throw an error.
    - **Solution**: You need to call the `contiguous()` method first.

- **reshape operation**:
    - **Flexible Handling**: The `reshape` operation is more flexible and does not require the tensor to be contiguous.
    - **Automatic Handling**: If the tensor is not contiguous, `reshape` will automatically create a new copy.
    - **Always Successful**: As long as the shapes match, the `reshape` operation will always succeed.

A key difference is that `reshape` can operate on non-contiguous Tensors by implicitly creating a new, contiguous one, whereas `view` requires the Tensor to be contiguous.

```python
import mindspore
import numpy as np
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ReshapeNet(nn.Cell):
    def construct(self, x):
        return x.reshape(3, 4)

class ViewNet(nn.Cell):
    def construct(self, x):
        return x.view(3, 4)

# Create a non-contiguous Tensor
a = mindspore.tensor(np.arange(12).reshape(3, 4))
b = a.transpose(1, 0) # b is now non-contiguous
print("b is contiguous:", b.is_contiguous())

# reshape can execute successfully
reshape_net = ReshapeNet()
c = reshape_net(b)
print("reshape success, c shape:", c.shape)

# view will fail
try:
    view_net = ViewNet()
    d = view_net(b)
except RuntimeError as e:
    print("view failed:", e)
```

The running result is as follows:

```text
b is contiguous: False
reshape success, c shape: (3, 4)
view failed: The tensor is not contiguous. You can call .contiguous() to get a contiguous tensor.
```

## Comparison of View and Inplace Features

### View Features

- **Shared Data**: The new Tensor created by the View operator shares the underlying data with the original Tensor and does not copy the data.
- **New Tensor Object**: The View operation returns a new Tensor object, but this new object points to the original data memory.
- **Independent Properties**: The new Tensor can have independent shape, stride, and dtype.
- **Bidirectional Impact**: Modifying the data of the View Tensor affects the original Tensor, and vice versa.

### Inplace Features

- **In-place Modification**: The Inplace operator directly calculates and modifies the data on the memory of the original Tensor without creating a new Tensor object.
- **Memory Saving**: Since no new Tensor is created to store the result, memory can be saved.
- **Naming Convention**: In MindSpore, Inplace operators are usually identified by a trailing underscore `_`, such as `add_`.

### Main Differences

| Feature | View | Inplace |
| :--- | :--- | :--- |
| **Return Value** | Returns a new Tensor object | Returns the modified original Tensor object |
| **Number of Objects** | Creates a new Tensor object that shares data with the original object | Does not create a new object, modifies it on the original object |
| **Core Purpose** | Efficiently access data from different "perspectives" | Save memory, calculate and update directly on the original data |

For more information on the usage of view inplace features, please refer to the following document:
Reference [view inplace](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/compile/static_graph.md#view-and-in-place-operations)
