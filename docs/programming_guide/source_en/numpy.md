# Numpy Interfaces in MindSpore

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_en/numpy.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

MindSpore Numpy package contains a set of Numpy-like interfaces, which allows developers to build models on MindSpore with similar syntax of Numpy.

## Operator Functions

Mindspore Numpy operators can be classified into four functional modules: `array generation`, `array operation`, `logic operation` and `math operation`. For details about the supported operators on the Ascend AI processors, GPU, and CPU, see [Numpy Interface List](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.numpy.html).

### Array Generations

Array generation operators are used to generate tensors.

Here is an example to generate an array:

```python
import mindspore.numpy as np
import mindspore.ops as ops
input_x = np.array([1, 2, 3], np.float32)
print("input_x =", input_x)
print("type of input_x =", ops.typeof(input_x))
```

The output is:

```python
input_x = [1. 2. 3.]
type of input_x = Tensor[Float32]
```

Here we have more examples:

#### Generate a tensor filled with the same element

`np.full` can be used to generate a tensor with user-specified values:

```python
import mindspore.numpy as np
input_x = np.full((2, 3), 6, np.float32)
print(input_x)
```

The output is:

```python
[[6. 6. 6.]
 [6. 6. 6.]]
```

Here is another example to generate an array with the specified shape and filled with the value of 1:

```python
import mindspore.numpy as np
input_x = np.ones((2, 3), np.float32)
print(input_x)
```

The output is:

```python
[[1. 1. 1.]
 [1. 1. 1.]]
```

#### Generate tensors in a specified range

Generate an arithmetic array within the specified range：

```python
import mindspore.numpy as np
input_x = np.arange(0, 5, 1)
print(input_x)
```

The output is:

```python
[0 1 2 3 4]
```

#### Generate tensors with specific requirement

Generate a matrix where the lower elements are 1 and the upper elements are 0 on the given diagonal:

```python
import mindspore.numpy as np
input_x = np.tri(3, 3, 1)
print(input_x)
```

The output is:

```python
[[1. 1. 0.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

Another example, generate a 2-D matrix with a diagonal of 1 and other elements of 0:

```python
import mindspore.numpy as np
input_x = np.eye(2, 2)
print(input_x)
```

The output is:

```python
[[1. 0.]
 [0. 1.]]
```

### Array Operations

Array operations focus on tensor manipulation.

#### Manipulate the shape of the tensor

For example, transpose a matrix:

```python
import mindspore.numpy as np
input_x = np.arange(10).reshape(5, 2)
output = np.transpose(input_x)
print(output)
```

The output is:

```python
[[0 2 4 6 8]
 [1 3 5 7 9]]
```

Another example, swap two axes:

```python
import mindspore.numpy as np
input_x = np.ones((1, 2, 3))
output = np.swapaxes(input_x, 0, 1)
print(output.shape)
```

The output is:

```python
(2, 1, 3)
```

#### Tensor splitting

Divide the input tensor into multiple tensors equally, for example:

```python
import mindspore.numpy as np
input_x = np.arange(9)
output = np.split(input_x, 3)
print(output)
```

The output is:

```python
(Tensor(shape=[3], dtype=Int32, value= [0, 1, 2]),
 Tensor(shape=[3], dtype=Int32, value= [3, 4, 5]),
 Tensor(shape=[3], dtype=Int32, value= [6, 7, 8]))
```

#### Tensor combination

Concatenate the two tensors according to the specified axis, for example:

```python
import mindspore.numpy as np
input_x = np.arange(0, 5)
input_y = np.arange(10, 15)
output = np.concatenate((input_x, input_y), axis=0)
print(output)
```

The output is:

```python
[ 0  1  2  3  4 10 11 12 13 14]
```

### Logic Operations

Logic operations define computations related with boolean types.
Examples of `equal` and `less` operations are as follows:

```python
import mindspore.numpy as np
input_x = np.arange(0, 5)
input_y = np.arange(0, 10, 2)
output = np.equal(input_x, input_y)
print("output of equal:", output)
output = np.less(input_x, input_y)
print("output of less:", output)
```

The output is:

```python
output of equal: [ True False False False False]
output of less: [False  True  True  True  True]
```

### Math Operations

Math operations include basic and advanced math operations on tensors, and they have full support on Numpy broadcasting rules. Here are some examples:

#### Sum two tensors

The following code implements the operation of adding two tensors of `input_x` and `input_y`:

```python
import mindspore.numpy as np
input_x = np.full((3, 2), [1, 2])
input_y = np.full((3, 2), [3, 4])
output = np.add(input_x, input_y)
print(output)
```

The output is:

```python
[[4 6]
 [4 6]
 [4 6]]
```

#### Matrics multiplication

The following code implements the operation of multiplying two matrices `input_x` and `input_y`:

```python
import mindspore.numpy as np
input_x = np.arange(2*3).reshape(2, 3).astype('float32')
input_y = np.arange(3*4).reshape(3, 4).astype('float32')
output = np.matmul(input_x, input_y)
print(output)
```

The output is:

```python
[[20. 23. 26. 29.]
 [56. 68. 80. 92.]]
```

#### Take the average along a given axis

The following code implements the operation of averaging all the elements of `input_x`:

```python
import mindspore.numpy as np
input_x = np.arange(6).astype('float32')
output = np.mean(input_x)
print(output)
```

The output is:

```python
2.5
```

#### Exponential arithmetic

The following code implements the operation of the natural constant `e` to the power of `input_x`:

```python
import mindspore.numpy as np
input_x = np.arange(5).astype('float32')
output = np.exp(input_x)
print(output)
```

The output is:

```python
[ 1.         2.718282   7.3890557 20.085537  54.598145 ]
```

## Interact With MindSpore Functions

Since `mindspore.numpy` directly wraps MindSpore tensors and operators, it has all the advantages and properties of MindSpore. In this section, we will briefly introduce how to employ MindSpore execution management and automatic differentiation in `mindspore.numpy` coding scenarios. These include:

- `ms_function`: for running codes in static graph mode for better efficiency.
- `GradOperation`: for automatic gradient computation.
- `mindspore.context`: for `mindspore.numpy` execution management.
- `mindspore.nn.Cell`: for using `mindspore.numpy` interfaces in MindSpore Deep Learning Models.

### Use ms_function to run code in static graph mode

Let's first see an example consisted of matrix multiplication and bias add, which is a typical process in Neural Networks:

```python
import mindspore.numpy as np

x = np.arange(8).reshape(2, 4).astype('float32')
w1 = np.ones((4, 8))
b1 = np.zeros((8,))
w2 = np.ones((8, 16))
b2 = np.zeros((16,))
w3 = np.ones((16, 4))
b3 = np.zeros((4,))

def forward(x, w1, b1, w2, b2, w3, b3):
    x = np.dot(x, w1) + b1
    x = np.dot(x, w2) + b2
    x = np.dot(x, w3) + b3
    return x

print(forward(x, w1, b1, w2, b2, w3, b3))
```

The output is:

```python
[[ 768.  768.  768.  768.]
 [2816. 2816. 2816. 2816.]]
```

In this function, MindSpore dispatches each computing kernel to device separately. However, with the help of `ms_function`, we can compile all operations into a single static computing graph.

```python
from mindspore import ms_function

forward_compiled = ms_function(forward)
```

> Currently, static graph cannot run in command line mode and not all python types can be passed into functions decorated with `ms_function`. For details about the static graph syntax support, see [Syntax Support](https://www.mindspore.cn/doc/note/en/r1.2/static_graph_syntax_support.html). For details about how to use `ms_function`, see [API: ms_function](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.html#mindspore.ms_function).

### Use GradOperation to compute deratives

`GradOperation` can be used to take deratives from normal functions and functions decorated with `ms_function`. Take the previous example:

```python
from mindspore import ops

grad_all = ops.composite.GradOperation(get_all=True)
grad_all(forward)(x, w1, b1, w2, b2, w3, b3)
```

To take the gradient of `ms_function` compiled functions, first we need to set the execution mode to static graph mode.

```python
from mindspore import ops, ms_function, context

context.set_context(mode=context.GRAPH_MODE)

grad_all = ops.composite.GradOperation(get_all=True)
grad_all(ms_function(forward))(x, w1, b1, w2, b2, w3, b3)
```

 For more details, see [API: GradOperation](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/ops/mindspore.ops.GradOperation.html).

### Use mindspore.context to control execution mode

Most functions in `mindspore.numpy` can run in Graph Mode and PyNative Mode, and can run on `CPU`，`GPU` and `Ascend`. Like MindSpore, users can manage the execution mode using `mindspore.context`：

```python
import mindspore.numpy as np
from mindspore import context

# Execution in static graph mode
context.set_context(mode=context.GRAPH_MODE)

# Execution in dynamic graph mode
context.set_context(mode=context.PYNATIVE_MODE)

# Execution on CPU backend
context.set_context(device_target="CPU")

# Execution on GPU backend
context.set_context(device_target="GPU")

# Execution on Ascend backend
context.set_context(device_target="Ascend")
...
```

 For more details, see [API: mindspore.context](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.context.html).

### Use mindspore.numpy in MindSpore Deep Learning Models

`mindspore.numpy` interfaces can be used inside `nn.cell` blocks as well. For example, the above code can be modified to:

```python
import mindspore.numpy as np
from mindspore import context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)

x = np.arange(8).reshape(2, 4).astype('float32')
w1 = np.ones((4, 8))
b1 = np.zeros((8,))
w2 = np.ones((8, 16))
b2 = np.zeros((16,))
w3 = np.ones((16, 4))
b3 = np.zeros((4,))

class NeuralNetwork(Cell):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    def construct(self, x, w1, b1, w2, b2, w3, b3):
        x = np.dot(x, w1) + b1
        x = np.dot(x, w2) + b2
        x = np.dot(x, w3) + b3
        return x

net = NeuralNetwork()

print(net(x, w1, b1, w2, b2, w3, b3))
```

The output is:

```python
[[ 768.  768.  768.  768.]
 [2816. 2816. 2816. 2816.]]
```

For more details on building Neural Network with MindSpore, see [MindSpore Training Guide](https://www.mindspore.cn/tutorial/training/en/r1.2/index.html).
