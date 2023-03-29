# Applying PyNative Mode

<a href="https://gitee.com/mindspore/docs/blob/r2.0/tutorials/experts/source_en/debug/pynative.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

In PyNative mode, MindSpore supports the execution of single operators, ordinary functions and networks, as well as the operation of individual gradients. Below we will introduce the use of these operations and considerations in detail through sample code.

## Executing Operations

First, we import the dependencies and set the run mode to PyNative mode:

```python
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)
```

### Executing Single Operators

The following is example code of executing Add operator [mindspore.ops.Add](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.Add.html#mindspore.ops.Add):

```python
add = ops.Add()
x = ms.Tensor(np.array([1, 2]).astype(np.float32))
y = ms.Tensor(np.array([3, 5]).astype(np.float32))
z = add(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

### Executing Function

Execute the custom function `add_func`. The sample code is as follows:

```python
add = ops.Add()

def add_func(x, y):
    z = add(x, y)
    z = add(z, x)
    return z

x = ms.Tensor(np.array([1, 2]).astype(np.float32))
y = ms.Tensor(np.array([3, 5]).astype(np.float32))
z = add_func(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

### Executing Network

Execute a custom network `Net` to define the network structure in the construst, and the sample code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

net = Net()
x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
z = net(x, y)

print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

## Synchronous Execution

In PyNative mode, in order to improve performance, the operator uses asynchronous execution on the device, so when the operator executes incorrectly, the error message may not be displayed until the program is executed until the end. In response to this situation, MindSpore added a pynative_synchronize setting to control whether asynchronous execution is used on the operator device.

In PyNative mode, the operator defaults to asynchronous execution, and you can control whether the execution is asynchronous by setting the content. When operator execution fails, it is convenient to see the location of the code where the error occurred through the calling stack. The sample code is as follows:

```python
import mindspore as ms

# Synchronize operator execution by setting the pynative_synchronize
ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = ops.GetNext([ms.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

ms.set_context()
x1 = np.random.randn(1, 1).astype(np.float32)
net = Net()
output = net(ms.Tensor(x1))
print(output.asnumpy())
```

Output: At this time, the operator is synchronous execution, and when the operator executes incorrectly, you can see the complete call stack and find the wrong line of code.

```text
Traceback (most recent call last):
  File "test.py", line 24, in <module>
    output = net(Tensor(x1))
  File ".../mindspore/nn/cell.py", line 602, in __call__
    raise err
  File ".../mindspore/nn/cell.py", line 599, in __call__
    output = self._run_construct(cast_inputs, kwargs)
  File ".../mindspore/nn/cell.py", line 429, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "test.py", line 17, in construct
    x = self.get_next()
  File ".../mindspore/ops/primitive.py", line 294, in __call__
    return _run_op(self, self.name, args)
  File ".../mindspore/common/api.py", line 90, in wrapper
    results = fn(*arg, **kwargs)
  File ".../mindspore/ops/primitive.py", line 754, in _run_op
    output = real_run_op(obj, op_name, args)
RuntimeError: mindspore/ccsrc/plugin/device/gpu/kernel/data/dataset_iterator_kernel.cc:139 Launch] For 'GetNext', gpu Queue(test) Open Failed: 2
```

## Hook Function

Debugging deep learning networks is a big task for every practitioner in the field of deep learning. Since the deep learning network hides the input and output data as well as the inverse gradient of the intermediate layer operators, only the gradient of the network input data (feature quantity and weight) is provided, resulting in the inability to accurately sense the data changes of the intermediate layer operators, which reduces the debugging efficiency. In order to facilitate users to debug the deep learning network accurately and quickly, MindSpore designes Hook function in dynamic graph mode. **Using Hook function can capture the input and output data of intermediate layer operators as well as the reverse gradient**.

Currently, four forms of Hook functions are provided in dynamic graph mode: HookBackward operator and register_forward_pre_hook, register_forward_hook, register_backward_hook functions registered on Cell objects.

### HookBackward Operator

HookBackward implements the Hook function in the form of an operator. The user initializes a HookBackward operator and places it at the location in the deep learning network where the gradient needs to be captured. In the forward execution of the network, the HookBackward operator outputs the input data as is without any modification. When the network back propagates the gradient, the Hook function registered on HookBackward will capture the gradient back propagated to this point. The user can customize the operation on the gradient in the Hook function, such as printing the gradient, or returning a new gradient.

The sample code is as follows:

```python
import mindspore as ms
from mindspore import ops

ms.set_context(mode=ms.PYNATIVE_MODE)

def hook_fn(grad_out):
    """Print Gradient"""
    print("hook_fn print grad_out:", grad_out)

hook = ops.HookBackward(hook_fn)
def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def net(x, y):
    return ms.grad(hook_test, grad_position=(0, 1))(x, y)

output = net(ms.Tensor(1, ms.float32), ms.Tensor(2, ms.float32))
print("output:", output)
```

```text
hook_fn print grad_out: (Tensor(shape=[], dtype=Float32, value= 2),)
output: (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
```

For more descriptions of the HookBackward operator, refer to the [API documentation](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.HookBackward.html).

### register_forward_pre_hook Function in Cell Object

The user can use the `register_forward_pre_hook` function on the Cell object to register a custom Hook function to capture data that is passed to that Cell object. This function does not work in static graph mode and inside functions modified with `@jit`. The `register_forward_pre_hook` function takes the Hook function as an input and returns a `handle` object that corresponds to the Hook function. The user can remove the corresponding Hook function by calling the `remove()` function of the `handle` object. Each call to the `register_forward_pre_hook` function returns a different `handle` object. Hook functions should be defined in the following way.

```python
def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
```

Here cell_id is the name of the Cell object as well as the ID information, and inputs are the data passed forward to the Cell object. Therefore, the user can use the register_forward_pre_hook function to capture the positive input data of a particular Cell object in the network. The user can customize the operations on the input data in the Hook function, such as viewing, printing data, or returning new input data to the current Cell object. If the original input data of the Cell object is computed in the Hook function and then returned as new input data, these additional computation operations will act on the backpropagation of the gradient at the same time.

The sample code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    input_x = inputs[0]
    return input_x

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.handle = self.relu.register_forward_pre_hook(forward_pre_hook_fn)

    def construct(self, x, y):
        x = x + y
        x = self.relu(x)
        return x

net = Net()
grad_net = ms.grad(net, grad_position=(0, 1))

x = ms.Tensor(np.ones([1]).astype(np.float32))
y = ms.Tensor(np.ones([1]).astype(np.float32))

output = net(x, y)
print(output)
gradient = grad_net(x, y)
print(gradient)
net.handle.remove()
gradient = grad_net(x, y)
print(gradient)
```

```text
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
[2.]
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
(Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
```

If the user returns the newly created data directly in the Hook function, instead of returning the data obtained from the original input data after calculation, then the back propagation of the gradient will be cut off on that Cell object.

The sample code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    return ms.Tensor(np.ones([1]).astype(np.float32))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.handle = self.relu.register_forward_pre_hook(forward_pre_hook_fn)

    def construct(self, x, y):
        x = x + y
        x = self.relu(x)
        return x

net = Net()
grad_net = ms.grad(net, grad_position=(0, 1))

x = ms.Tensor(np.ones([1]).astype(np.float32))
y = ms.Tensor(np.ones([1]).astype(np.float32))

gradient = grad_net(x, y)
print(gradient)
```

```text
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
(Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]))
```

To avoid running failure when scripts switch to graph mode, it is not recommended to call the `register_forward_pre_hook` function and the `remove()` function of the `handle` object in the `construct` function of the Cell object. In dynamic graph mode, if the `register_forward_pre_hook` function is called in the `construct` function of the Cell object, the Cell object will register a new Hook function every time it runs.

For more information about the `register_forward_pre_hook` function of the Cell object, refer to the [API documentation](https://mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook).

### register_forward_hook Function of Cell Object

The user can use the `register_forward_hook` function on the Cell object to register a custom Hook function that captures the data passed forward to the Cell object and the output data of the Cell object. This function does not work in static graph mode and inside functions modified with `@jit`. The `register_forward_hook` function takes the Hook function as an input and returns a `handle` object that corresponds to the Hook function. The user can remove the corresponding Hook function by calling the `remove()` function of the `handle` object. Each call to the `register_forward_hook` function returns a different `handle` object. Hook functions should be defined in the following way.

The sample code is as follows:

```python
def forward_hook_fn(cell_id, inputs, outputs):
    print("forward inputs: ", inputs)
    print("forward outputs: ", outputs)
```

Here `cell_id` is the name of the Cell object and the ID information, `inputs` is the forward input data to the Cell object, and `outputs` is the forward output data of the Cell object. Therefore, the user can use the `register_forward_hook` function to capture the forward input data and output data of a particular Cell object in the network. Users can customize the operations on input and output data in the Hook function, such as viewing, printing data, or returning new output data. If the original output data of the Cell object is computed in the Hook function and then returned as new output data, these additional computation operations will act on the back propagation of the gradient at the same time.

The sample code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE)

def forward_hook_fn(cell_id, inputs, outputs):
    print("forward inputs: ", inputs)
    print("forward outputs: ", outputs)
    outputs = outputs + outputs
    return outputs

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.handle = self.relu.register_forward_hook(forward_hook_fn)

    def construct(self, x, y):
        x = x + y
        x = self.relu(x)
        return x

net = Net()
grad_net = ms.grad(net, grad_position=(0, 1))

x = ms.Tensor(np.ones([1]).astype(np.float32))
y = ms.Tensor(np.ones([1]).astype(np.float32))

gradient = grad_net(x, y)
print(gradient)
net.handle.remove()
gradient = grad_net(x, y)
print(gradient)
```

```text
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
forward outputs:  [2.]
(Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]))
(Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
```

If the user returns the newly created data directly in the Hook function, instead of returning new output data that is obtained after the original output data is calculated, then the back propagation of the gradient will cut off on that Cell object, which can be seen in the use case illustration of the `register_forward_pre_hook` function.

To avoid running failure when the script switches to graph mode, it is not recommended to call the `register_forward_hook` function in the `construct` function of the Cell object and the `remove()` function of the `handle` object. In dynamic graph mode, if the `register_forward_hook` function is called in the `construct` function of the Cell object, the Cell object will register a new Hook function every time it runs.

For more information about the `register_forward_hook` function of the Cell object, please refer to the [API documentation](https://mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook).

### register_backward_hook Function of Cell Object

The user can use the `register_backward_hook` function on the Cell object to register a custom Hook function that captures the gradient associated with the Cell object when the network is back propagated. This function does not work in graph mode or inside functions modified with `@jit`. The `register_backward_hook` function takes the Hook function as an input and returns a `handle` object that corresponds to the Hook function. The user can remove the corresponding Hook function by calling the `remove()` function of the `handle` object. Each call to the `register_backward_hook` function will return a different `handle` object.

Unlike the custom Hook function used by the HookBackward operator, the inputs of the Hook function used by `register_backward_hook` contains `cell_id`, which represents the name and id information of the Cell object, the gradient passed to the Cell object in reverse, and the gradient of the reverse output of the Cell object.

The sample code is as follows:

```python
def backward_hook_function(cell_id, grad_input, grad_output):
    print(grad_input)
    print(grad_output)
```

Here `cell_id` is the name and the ID information of the Cell object, `grad_input` is the gradient passed to the Cell object when the network is back-propagated, which corresponds to the reverse output gradient of the next operator in the forward process. `grad_output` is the gradient of the reverse output of the Cell object. Therefore, the user can use the `register_backward_hook` function to capture the backward input and backward output gradients of a particular Cell object in the network. The user can customize the operations on the gradient in the Hook function, such as viewing, printing the gradient, or returning the new output gradient. If you need to return the new output gradient in the Hook function, the return value must be in the form of `tuple`.

The sample code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE)

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
grad_net = ms.grad(net)
output = grad_net(ms.Tensor(np.ones([1, 1, 2, 2]).astype(np.float32)))
print(output)
net.handle.remove()
output = grad_net(ms.Tensor(np.ones([1, 1, 2, 2]).astype(np.float32)))
print("-------------\n", output)
```

```text
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 1.00000000e+00]],
  [[ 1.00000000e+00]]]]),)
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 9.99994993e-01]],
  [[ 9.99994993e-01]]]]),)
(Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
[[[[ 1.99998999e+00,  1.99998999e+00],
   [ 1.99998999e+00,  1.99998999e+00]]]]),)
-------------
 (Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
[[[[ 1.99998999e+00,  1.99998999e+00],
   [ 1.99998999e+00,  1.99998999e+00]]]]),)
```

When the `register_backward_hook` function and the `register_forward_pre_hook` function, and the `register_forward_hook` function act on the same Cell object at the same time, if the `register_forward_pre_hook` and the `register_forward_hook` functions add other operators for data processing, these new operators will participate in the forward calculation of the data before or after the execution of the Cell object, but the backward gradient of these new operators is not captured by the `register_backward_hook` function. The Hook function registered in `register_backward_hook` only captures the input and output gradients of the original Cell object.

The sample code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE)

def forward_pre_hook_fn(cell_id, inputs):
    print("forward inputs: ", inputs)
    input_x = inputs[0]
    return input_x

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
        self.relu = nn.ReLU()
        self.handle = self.relu.register_forward_pre_hook(forward_pre_hook_fn)
        self.handle2 = self.relu.register_forward_hook(forward_hook_fn)
        self.handle3 = self.relu.register_backward_hook(backward_hook_fn)

    def construct(self, x, y):
        x = x + y
        x = self.relu(x)
        return x

net = Net()
grad_net = ms.grad(net, grad_position=(0, 1))
gradient = grad_net(ms.Tensor(np.ones([1]).astype(np.float32)), ms.Tensor(np.ones([1]).astype(np.float32)))
print(gradient)
```

```text
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
forward inputs:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
forward outputs:  [2.]
grad input:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
grad output:  (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
(Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]))
```

Here `grad_input` is the gradient passed to `self.relu` when the gradient is back-propagated, not the gradient of the new `Add` operator in the `forward_hook_fn` function. Here `grad_output` is the reverse output gradient of the `self.relu` when the gradient is back-propagated, not the reverse output gradient of the new `Add` operator in the `forward_pre_hook_fn` function. The `register_forward_pre_hook` and `register_forward_hook` functions work before and after the execution of the Cell object and do not affect the gradient capture range of the reverse Hook function on the Cell object.

To avoid running failure when the scripts switch to graph mode, it is not recommended to call the `register_backward_hook` function and the `remove()` function of the `handle` object in the `construct` function of the Cell object. In PyNative mode, if the `register_backward_hook` function is called in the `construct` function of the Cell object, the Cell object will register a new Hook function every time it runs.

For more information about the `register_backward_hook` function of the Cell object, please refer to the [API documentation](https://mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook).

## Customizing bprop Function

The user can customize the back-propagation (calculation) function of the nn.Cell object, thus controlling the process of gradient calculation of the nn.Cell object and locating the gradient problem. The custom bprop function is used by adding a user-defined bprop function inside the defined nn.Cell object. The user-defined bprop function is used to generate the inverse map during the training process.

The sample code is as follows:

```python
import mindspore.nn as nn
import mindspore as ms


ms.set_context(mode=ms.PYNATIVE_MODE)

class Net(nn.Cell):
    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout

grad_net = ms.grad(Net(), grad_position=(0, 1))
output = grad_net(ms.Tensor(1, ms.float32), ms.Tensor(2, ms.float32))
print(output)
```

```text
(Tensor(shape=[], dtype=Float32, value= 3), Tensor(shape=[], dtype=Float32, value= 2))
```

