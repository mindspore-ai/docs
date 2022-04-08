# Debugging in PyNative Mode

`Ascend` `GPU` `CPU` `Model Running`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/debug_in_pynative_mode.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore supports the following running modes which are optimized for debugging or running:

- PyNative mode: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.
- Graph mode: static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution. This mode uses technologies such as graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

By default, MindSpore is in Graph mode. You can switch it to PyNative mode by calling `context.set_context(mode=context.PYNATIVE_MODE)`. Similarly, MindSpore in PyNative mode can be switched to Graph mode through `context.set_context(mode=context.GRAPH_MODE)`.

In PyNative mode, single operators, common functions, network inference, and separated gradient calculation can be executed. The following describes the usage and precautions.

> In PyNative mode, operators are executed asynchronously on the device to improve performance. Therefore, when an error occurs during operator execution, the error information may be displayed after the program is executed. Therefore, in PyNative mode, a pynative_synchronize setting is added to control whether operators are executed asynchronously on the device.
>
> In the following example, the parameter initialization uses random values, and the output results in specific execution may be different from the results of local execution; if you need to stabilize the output of a fixed value, you can set a fixed random seed. For the setting method, please refer to [mindspore.set_seed()](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.set_seed.html).

## Executing a Single Operator

Execute a single operator and output the result, as shown in the following example.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
output = conv(input_data)
print(output.asnumpy())
```

Output:

```text
[[[[-0.02190447 -0.05208071 -0.05208071 -0.05208071 -0.06265172]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01529094 -0.05286242 -0.05286242 -0.05286242 -0.04228776]
[-0.01430791 -0.04892948 -0.04892948 -0.04892948 -0.01096004]]

[[ 0.00802889 -0.00229866 -0.00229866 -0.00229866 -0.00471579]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01172971 0.02172665 0.02172665 0.02172665 0.03261888]
[ 0.01784375 0.01185635 0.01185635 0.01185635 0.01839031]]

[[ 0.04841832 0.03321705 0.03321705 0.03321705 0.0342317 ]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.0651359 0.04310361 0.04310361 0.04310361 0.03355784]
[ 0.04680437 0.03465693 0.03465693 0.03465693 0.00171057]]

[[-0.01783456 -0.00459451 -0.00459451 -0.00459451 0.02316688]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.01295831 0.00879035 0.00879035 0.00879035 0.01178642]
[ 0.05016355 0.03958241 0.03958241 0.03958241 0.03443141]]]]
```

## Executing a Common Function

Combine multiple operators into a function, call the function to execute the operators, and output the result, as shown in the following example:

Example Code:

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def add_func(x, y):
    z = ops.add(x, y)
    z = ops.add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = add_func(x, y)
print(output.asnumpy())
```

Output:

```text
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

> Summary is not supported in PyNative mode, so summary related operators cannot be used.

### Improving PyNative Performance

MindSpore provides the Staging function to improve the execution speed of inference tasks in PyNative mode. This function compiles Python functions or Python class methods into computational graphs in PyNative mode and improves the execution speed by using graph optimization technologies, as shown in the following example:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(x, y) # Staging mode
add = ops.Add()
res = add(x, z) # PyNative mode
print(res.asnumpy())
```

Output:

```text
[[3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]]
```

In the preceding code, the `ms_function` decorator is added before `construct` of the `TensorAddNet` class. The decorator compiles the `construct` method into a computational graph. After the input is given, the graph is delivered and executed, `add` in the preceding code is executed in the common PyNative mode.

It should be noted that, in a function to which the `ms_function` decorator is added, if an operator (such as `pooling` or `add`) that does not need parameter training is included, the operator can be directly called in the decorated function, as shown in the following example:

Example Code:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

add = ops.Add()

@ms_function
def add_fn(x, y):
    res = add(x, y)
    return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
z = add_fn(x, y)
print(z.asnumpy())
```

Output:

```text
[[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

If the decorated function contains operators (such as `Convolution` and `BatchNorm`) that require parameter training, these operators must be instantiated before the decorated function is called, as shown in the following example:

Example Code:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
conv_obj.init_parameters_data()
@ms_function
def conv_fn(x):
    res = conv_obj(x)
    return res

input_data = np.random.randn(2, 3, 6, 6).astype(np.float32)
z = conv_fn(Tensor(input_data))
print(z.asnumpy())
```

Output:

```text
[[[[ 0.10377571 -0.0182163 -0.05221086]
[ 0.1428334 -0.01216263 0.03171652]
[-0.00673915 -0.01216291 0.02872104]]

[[ 0.02906547 -0.02333629 -0.0358406 ]
[ 0.03805163 -0.00589525 0.04790922]
[-0.01307234 -0.00916951 0.02396654]]

[[ 0.01477884 -0.06549098 -0.01571796]
[ 0.00526886 -0.09617482 0.04676902]
[-0.02132788 -0.04203424 0.04523344]]

[[ 0.04590619 -0.00251453 -0.00782715]
[ 0.06099087 -0.03445276 0.00022781]
[ 0.0563223 -0.04832596 -0.00948266]]]

[[[ 0.08444098 -0.05898955 -0.039262 ]
[ 0.08322686 -0.0074796 0.0411371 ]
[-0.02319113 0.02128408 -0.01493311]]

[[ 0.02473745 -0.02558945 -0.0337843 ]
[-0.03617039 -0.05027632 -0.04603915]
[ 0.03672804 0.00507637 -0.08433761]]

[[ 0.09628943 0.01895323 -0.02196114]
[ 0.04779419 -0.0871575 0.0055248 ]
[-0.04382382 -0.00511185 -0.01168541]]

[[ 0.0534859 0.02526264 0.04755395]
[-0.03438103 -0.05877855 0.06530266]
[ 0.0377498 -0.06117418 0.00546303]]]]
```

## Debugging Network Train Model

In PyNative mode, the gradient can be calculated separately. As shown in the following example, `GradOperation` is used to calculate all input gradients of the function or the network. Note that the inputs have to be Tensor.

Example Code:

```python
import mindspore.ops as ops
import mindspore.context as context
from mindspore import dtype as mstype
from mindspore import Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def mul(x, y):
    return x * y

def mainf(x, y):
    return ops.GradOperation(get_all=True)(mul)(x, y)

print(mainf(Tensor(1, mstype.int32), Tensor(2, mstype.int32)))
```

Output:

```text
(Tensor(shape=[], dtype=Int32, value=2), Tensor(shape=[], dtype=Int32, value=1))
```

During network training, obtain the gradient, call the optimizer to optimize parameters (the breakpoint cannot be set during the reverse gradient calculation), and calculate the loss values. Then, network training is implemented in PyNative mode.

Complete LeNet Sample Code:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import Dense, WithLossCell, SoftmaxCrossEntropyWithLogits, Momentum

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return ops.GradOperation(get_by_list=True)(self.network, weights)(x, label)

net = LeNet5()
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
train_network = GradWrap(net_with_criterion)
train_network.set_train()

input_data = Tensor(np.ones([net.batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.ones([net.batch_size]).astype(np.int32))
output = net(Tensor(input_data))
loss_output = criterion(output, label)
grads = train_network(input_data, label)
success = optimizer(grads)
loss = loss_output.asnumpy()
print(loss)
```

Output:

```text
2.3050091
```

In the preceding execution, an intermediate result of network execution can be obtained at any required place in `construt` function, and the network can be debugged by using the Python Debugger (pdb).

## Synchronous Execution Under PyNative

In PyNative mode, the operators are executed asynchronously by default. You can control whether to execute asynchronously by setting the context. When the operator fails to execute, you can easily see the error code location through the call stack.

Set context pynative_synchronize to True：

```python
context.set_context(pynative_synchronize=True)
```

Example Code:

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = ops.GetNext([mstype.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

context.set_context()
x1 = np.random.randn(1, 1).astype(np.float32)
net = Net()
output = net(Tensor(x1))
print(output.asnumpy())
```

Output: you can see the complete call stack.

```text
Traceback (most recent call last):
  File "test_pynative_sync_control.py", line 41, in <module>
    output = net(Tensor(x1))
  File "mindspore/mindspore/nn/cell.py", line 406, in <module>
    output = self.run_construct(cast_inputs, kwargs)
  File "mindspore/mindspore/nn/cell.py", line 348, in <module>
    output = self.construct(*cast_inputs, **kwargs)
  File "test_pynative_sync_control.py", line 33, in <module>
    x = self.get_next()
  File "mindspore/mindspore/ops/primitive.py", line 247, in <module>
    return _run_op(self, self.name, args)
  File "mindspore/mindspore/common/api.py", line 77, in <module>
    results = fn(*arg, **kwargs)
  File "mindspore/mindspore/ops/primitive.py", line 677, in _run_op
    output = real_run_op(obj, op_name, args)
RuntimeError: mindspore/ccsrc/runtime/device/kernel_runtime.cc:1006 DebugStreamSync] Op Default/GetNext-op0 run failed!
```

## Hook

Debugging deep learning network is a task that practitioners in every field of deep learning need to face and invest a lot of energy. Because the deep learning network hides the input, output data and gradients of the middle layer operator and only provides the gradients of the network input data (feature data and weight), developers can not accurately perceive the data changes of the middle layer operator, which affects the debugging efficiency. In order to facilitate developers to accurately and quickly debug the deep learning network, MindSpore designed the hook function in PyNative mode. Developers can use the hook function to capture the input, output data and gradients of the middle layer operator. At present, PyNative mode provides four forms of hook functions: HookBackward operator and the register_forward_pre_hook function, the register_forward_hook function, the register_backward_hook function for Cell object.

### HookBackward operator

HookBackward implements the hook function as an operator. The user initializes a HookBackward operator and inserts it into the position where the gradient needs to be captured in the deep learning network. When the network is executing forward process, the HookBackward operator outputs the input data as it is without any modification; When the network back propagates gradient, the hook function registered on HookBackward operator will capture the gradient back propagated to this point. You can customize the gradient operation in the hook function, such as printing the gradient or returning a new gradient.

Example Code:

```python
import mindspore
from mindspore import ops
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def hook_fn(grad_out):
    print(grad_out)

grad_all = GradOperation(get_all=True)
hook = ops.HookBackward(hook_fn)
def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def net(x, y):
    return grad_all(hook_test)(x, y)

output = net(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
print(output)
```

Output：

```python
(Tensor(shape=[], dtype=Float32, value= 2),)
(Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
```

For more descriptions of HookBackward operator, please refer to [API document](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.HookBackward.html).

### The register_forward_pre_hook function for Cell object

Users can register a custom hook function by using the `register_forward_pre_hook` function on the Cell object, which is used to capture the forward data passed to this Cell object. This function does not work in graph mode or on Cell object decorated with `ms_function` . The `register_forward_pre_hook` function receives an user-defined hook function as the input parameter and returns a `handle` object corresponding to the hook function. Users can delete the corresponding hook function by calling the `remove()` function of the `handle` object. Each time the `register_forward_pre_hook` function is called, a different `handle` object is returned. The hook function should be defined as follows.

Example Code:

```python
def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
```

The `cell_id` is the name and ID information of the Cell object, the `inputs` are the forward data passed to the Cell object. Therefore, users can use the `register_forward_pre_hook` function to capture the forward input data of a Cell object in the network. You can customize the operation of input data in the hook function, such as checking data, printing data, or returning new input data to the current Cell object. If the original input data of the Cell object is calculated in the hook function and then returned as new input data, these new calculation operations will act on the back propagation of the gradient.

Example Code:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    input_x = inputs[0] + inputs[1]
    return input_x, inputs[1]

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_forward_pre_hook(forward_pre_hook_fn)

    def construct(self, x, y):
        x = x + x
        x = self.mul(x, y)
        return x

grad = GradOperation(get_all=True)
net = Net()
output = net(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(output)
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
net.handle.remove()
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
```

Output：

```python
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
3.0
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 4.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]))
```

If user returns the newly created data directly in the hook function instead of returning the calculated data from the original input data, the back propagation of the gradient will be cut off on this Cell object.

Example Code:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    return Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_forward_pre_hook(forward_pre_hook_fn)

    def construct(self, x, y):
        x = x + x
        x = self.mul(x, y)
        return x

grad = GradOperation(get_all=True)
net = Net()
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
```

Output：

```python
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]))
```

In order to avoid script failure when switching to graph mode, it is not recommended to call the `register_forward_pre_hook` function or the `remove()` function of the `handle` object in the `construct` function of the Cell object. In the PyNative mode, if the `register_forward_pre_hook` function is called in the `construct` function of the Cell object, a hook function will be added at each run time of Cell object.

More about the `register_forward_pre_hook` interface, please refer to [API Document](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook).

### The register_forward_hook function for Cell object

Users can register a custom hook function by using the `register_forward_hook` function on the Cell object, which is used to capture the forward input and output data passed to this Cell object. This function does not work in graph mode or on Cell object decorated with `ms_function` . The `register_forward_hook` function receives an user-defined hook function as the input parameter and returns a `handle` object corresponding to the hook function. Users can delete the corresponding hook function by calling the `remove()` function of the `handle` object. Each time the `register_forward_hook` function is called, a different handle object is returned. The hook function should be defined as follows.

Example Code:

```python
def forward_hook_fn(cell_id, inputs, outputs):
    print("forward inputs: ", inputs)
    print("forward outputs: ", outputs)
```

The `cell_id` is the name and ID information of the Cell object, the `inputs` are the forward data passed to the Cell object, the `outputs` are the forward output data of the Cell object. Therefore, users can use the `register_forward_hook` function to capture the forward input and output data of a Cell object in the network. You can customize the operation of the input and output data in the hook function, such as checking data, printing data, or returning new output data. If the original output data of the Cell object is calculated in the hook function and then returned as new output data, these new calculation operations will act on the back propagation of the gradient.

Example Code:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def forward_hook_fn(cell_id, inputs, outputs):
    print("forward inputs: ", inputs)
    print("forward outputs: ", outputs)
    outputs = outputs + outputs
    return outputs

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_forward_hook(forward_hook_fn)

    def construct(self, x, y):
        x = x + x
        x = self.mul(x, y)
        return x

grad = GradOperation(get_all=True)
net = Net()
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
net.handle.remove()
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
```

Output：

```python
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
forward outputs: 2.0
(Tensor(shape=[1], dtype=Float32, value= [ 4.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 4.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]))
```

If user returns the newly created data directly in the hook function instead of returning the calculated data from the original output data, the back propagation of the gradient will be cut off on this Cell object. Refer to the same case description of the `register_forward_pre_hook` function for this phenomenon.

In order to avoid script failure when switching to graph mode, it is not recommended to call the `register_forward_hook` function or the `remove()` function of the `handle` object in the `construct` function of the Cell object. In the PyNative mode, if the `register_forward_hook` function is called in the `construct` function of the Cell object, a hook function will be added at each run time of Cell object.

More about the `register_forward_hook` interface, please refer to [API Document](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook).

### The register_backward_hook function for Cell object

Users can register a custom hook function by using the `register_backward_hook` function on the Cell object, which is used to capture the gradients associated with the Cell object during network back propagation. This function does not work in graph mode or on Cell object decorated with `ms_function` . The `register_backward_hook` function receives an user-defined hook function as the input parameter and returns a `handle` object corresponding to the hook function. Users can delete the corresponding hook function by calling the `remove()` function of the `handle` object. Each time the `register_backward_hook` function is called, a different `handle` object is returned.

Different from the HookBackward operator, the input parameters of the hook function registered in register_backward_hook function contain the `cell_id` representing name and ID information of the Cell object, the incoming gradient and the output gradient of Cell object.

Example Code:

```python
def backward_hook_function(cell_id, grad_input, grad_output):
    print(grad_input)
    print(grad_output)
```

The `cell_id` is the name and ID information of the Cell object. The `grad_input` is the input gradient of the Cell object, which corresponds to the output gradient of the next operator in the forward process. The `grad_output` is the output gradient of the Cell object. Therefore, users can use `register_backward_hook` interface to capture the input gradient and output gradient of the Cell object. Users can customize the gradient operation in the hook function, such as checking gradient, printing gradient or returning new output gradient. If users need to return new output gradient in the hook function, the return gradient must be in the form of `tuple` .

Example Code:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def backward_hook_function(cell_id, grad_input, grad_output):
    print(grad_input)
    print(grad_output)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.handle = self.bn.register_backward_hook(backward_hook_function)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

net = Net()
grad_all = GradOperation(get_all=True)
output = grad_all(net)(Tensor(np.ones([1, 1, 2, 2]).astype(np.float32)))
print(output)
net.handle.remove()
output = grad_all(net)(Tensor(np.ones([1, 1, 2, 2]).astype(np.float32)))
print(output)
```

Output：

```python
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 1.00000000e+00]],
  [[ 1.00000000e+00]]]]),)
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 9.99994993e-01]],
  [[ 9.99994993e-01]]]]),)
(Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
[[[[ 1.99998999e+00, 1.99998999e+00],
   [ 1.99998999e+00, 1.99998999e+00]]]]),)
(Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
[[[[ 1.99998999e+00, 1.99998999e+00],
   [ 1.99998999e+00, 1.99998999e+00]]]]),)
```

When `register_forward_pre_hook` function, `register_forward_hook` function and `register_backward_hook` function are registered on the same Cell object and new operators are added in hook function for data processing, these new operators will participate in the forward calculation before or after the execution of the Cell object, but the gradients of these new operators is not within the capture range of the backward hook function. The backward hook function registered by `register_backward_hook` interface only captures the input and output gradients of the original Cell object.

Example Code:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    input_x = inputs[0] + inputs[1]
    input_y = inputs[0] + inputs[1]
    return input_x, input_y

def forward_hook_fn(cell_id, inputs, outputs):
    print("forward inputs: ", inputs)
    print("forward outputs: ", outputs)
    outputs = outputs + outputs
    return outputs

def backward_hook_fn(cell_id, grad_input, grad_output):
    print("grad input: ", grad_input)
    print("grad output: ", grad_output)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_forward_pre_hook(forward_pre_hook_fn)
        self.handle2 = self.mul.register_forward_hook(forward_hook_fn)
        self.handle3 = self.mul.register_backward_hook(backward_hook_fn)

    def construct(self, x, y):
        x = x + x
        x = self.mul(x, y)
        return x

net = Net()
grad = GradOperation(get_all=True)
gradient = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
```

Output：

```python
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 3.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 3.00000000e+00]))
forward outputs: 9.0
grad input: (Tensor(shape=[], dtype=Float32, value= 2),)
grad output: (Tensor(shape=[1], dtype=Float32, value= [ 6.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 6.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 2.40000000e+01]), Tensor(shape=[1], dtype=Float32, value= [ 1.20000000e+01]))
```

The `grad input` is the input gradient of `self.mul` . It is not the input gradient of the `Add` operator in the `forward_hook_funcntion` . The `grad output` is the output gradient of `self.mul` . It is not the output gradient of the `Add` operators in the `forward_pre_hook_funcntion` . `register_forward_pre_hook` function and `register_forward_hook` function work before or after the execution of the Cell object. They will not affect the gradient capture range of the backward hook function on the Cell object.

In order to avoid script failure when switching to graph mode, it is not recommended to call the `register_backward_hook` function or the `remove()` function of the `handle` object in the `construct` function of the Cell object. In the PyNative mode, if the `register_backward_hook` function is called in the `construct` function of the Cell object, a hook function will be added at each run time of Cell object.

More about the `register_backward_hook` interface, please refer to [API Document](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook).

## Custom bprop

Users can customize the back propagation (calculation) function of the Cell object to control the gradient calculation process and positioning gradient problem. The custom bprop is implemented by defining a `bprop function` for Cell object. During the back propagation process, the custom bprop function will run.

Example Code:

```python
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout

grad_all = GradOperation(get_all=True)
output = grad_all(Net())(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
print(output)
```

Output：

```python
(Tensor(shape=[], dtype=Float32, value= 3), Tensor(shape=[], dtype=Float32, value= 2))
```
