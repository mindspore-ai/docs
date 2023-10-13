[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/source_en/advanced/modules/layer.md)

# Cell and Parameter

Cell, as the basic unit of neural network construction, corresponds to the concept of neural network layer, and the abstract encapsulation of Tensor computation operation can represent the neural network structure more accurately and clearly. In addition to the basic Tensor computation flow definition, the neural network layer contains functions such as parameter management and state management. Parameter is the core of neural network training and is usually used as an internal member variable of the neural network layer. In this section, we systematically introduce parameters, neural network layers and their related usage.

## Parameter

Parameter is a special class of Tensor, which is a variable whose value can be updated during model training. MindSpore provides the `mindspore.Parameter` class for Parameter construction. In order to distinguish between Parameter for different purposes, two different categories of Parameter are defined below. In order to distinguish between Parameter for different purposes, two different categories of Parameter are defined below:

- Trainable parameter. Tensor that is updated after the gradient is obtained according to the backward propagation algorithm during model training, and `required_grad` needs to be set to `True`.
- Untrainable parameters. Tensor that does not participate in backward propagation needs to update values (e.g. `mean` and `var` variables in BatchNorm), when `requires_grad` needs to be set to `False`.

> Parameter is set to `required_grad=True` by default.

We construct a simple fully-connected layer as follows:

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
        self.b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z

net = Network()
```

In the `__init__` method of `Cell`, we define two parameters `w` and `b` and configure `name` for namespace management. Use `self.attr` in the `construct` method to call directly to participate in Tensor operations.

### Obtaining Parameter

After constructing the neural network layer by using Cell+Parameter, we can use various methods to obtain the Parameter managed by Cell.

#### Obtaining a Single Parameter

To get a particular parameter individually, just call a member variable of a Python class directly.

```python
print(net.b.asnumpy())
```

```text
[-1.2192779  -0.36789745  0.0946381 ]
```

#### Obtaining a Trainable Parameter

Trainable parameters can be obtained by using the `Cell.trainable_params` method, and this interface is usually called when configuring the optimizer.

```python
print(net.trainable_params())
```

```text
[Parameter (name=w, shape=(5, 3), dtype=Float32, requires_grad=True), Parameter (name=b, shape=(3,), dtype=Float32, requires_grad=True)]
```

#### Obtaining All Parameters

Use the `Cell.get_parameters()` method to get all parameters, at which point a Python iterator will be returned.

```python
print(type(net.get_parameters()))
```

```text
<class 'generator'>
```

Or you can call `Cell.parameters_and_names` to return the parameter names and parameters.

```python
for name, param in net.parameters_and_names():
    print(f"{name}:\n{param.asnumpy()}")
```

```text
w:
[[ 4.15680408e-02 -1.20311625e-01  5.02573885e-02]
 [ 1.22175144e-04 -1.34980649e-01  1.17642188e+00]
 [ 7.57667869e-02 -1.74758151e-01 -5.19092619e-01]
 [-1.67846107e+00  3.27240258e-01 -2.06452996e-01]
 [ 5.72323874e-02 -8.27963874e-02  5.94243526e-01]]
b:
[-1.2192779  -0.36789745  0.0946381 ]
```

### Modifying the Parameter

#### Modifying Parameter Values Directly

Parameter is a special kind of Tensor, so its value can be modified by using the Tensor index modification.

```python
net.b[0] = 1.
print(net.b.asnumpy())
```

```text
[ 1.         -0.36789745  0.0946381 ]
```

#### Overriding the Modified Parameter Values

The `Parameter.set_data` method can be called to override the Parameter by using a Tensor with the same Shape. This method is commonly used for [Cell traversal initialization](https://www.mindspore.cn/tutorials/en/r2.2/advanced/modules/initializer.html) by using Initializer.

```python
net.b.set_data(Tensor([3, 4, 5]))
print(net.b.asnumpy())
```

```text
[3. 4. 5.]
```

#### Modifying Parameter Values During Runtime

The main role of parameters is to update their values during model training, which involves parameter modification during runtime after backward propagation to obtain gradients, or when untrainable parameters need to be updated. Due to the compiled design of MindSpore's [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.2/beginner/accelerate_with_static_graph.html), it is necessary at this point to use the `mindspore.ops.assign` interface to assign parameters. This method is commonly used in [Custom Optimizer](https://www.mindspore.cn/tutorials/en/r2.2/advanced/modules/optimizer.html) scenarios. The following is a simple sample modification of parameter values during runtime:

```python
import mindspore as ms

@ms.jit
def modify_parameter():
    b_hat = ms.Tensor([7, 8, 9])
    ops.assign(net.b, b_hat)
    return True

modify_parameter()
print(net.b.asnumpy())
```

```text
[7. 8. 9.]
```

### Parameter Tuple

ParameterTuple, variable tuple, used to store multiple Parameter, is inherited from tuple tuples, and provides cloning function.

The following example provides the ParameterTuple creation method:

```python
from mindspore.common.initializer import initializer
from mindspore import ParameterTuple
# Creation
x = Parameter(default_input=ms.Tensor(np.arange(2 * 3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], ms.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))

# Clone from params and change the name to "params_copy"
params_copy = params.clone("params_copy")

print(params)
print(params_copy)
```

```text
(Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))
(Parameter (name=params_copy.x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
```

## Cell Training State Change

Some Tensor operations in neural networks do not behave the same during training and inference, e.g., `nn.Dropout` performs random dropout during training but not during inference, and `nn.BatchNorm` requires updating the `mean` and `var` variables during training and fixing their values unchanged during inference. So we can set the state of the neural network through the `Cell.set_train` interface.

When `set_train` is set to True, the neural network state is `train`, and the default value of `set_train` interface is `True`:

```python
net.set_train()
print(net.phase)
```

```text
train
```

When `set_train` is set to False, the neural network state is `predict`:

```python
net.set_train(False)
print(net.phase)
```

```text
predict
```

## Custom Neural Network Layers

Normally, the neural network layer interface and function interface provided by MindSpore can meet the model construction requirements, but since the AI field is constantly updating, it is possible to encounter new network structures without built-in modules. At this point, we can customize the neural network layer through the function interface provided by MindSpore, Primitive operator, and can use the `Cell.bprop` method to customize the reverse. The following are the details of each of the three customization methods.

### Constructing Neural Network Layers by Using the Function Interface

MindSpore provides a large number of basic function interfaces, which can be used to construct complex Tensor operations, encapsulated as neural network layers. The following is an example of `Threshold` with the following equation:

$$
y =\begin{cases}
   x, &\text{ if } x > \text{threshold} \\
   \text{value}, &\text{ otherwise }
   \end{cases}
$$

It can be seen that `Threshold` determines whether the value of the Tensor is greater than the `threshold` value, keeps the value whose judgment result is `True`, and replaces the value whose judgment result is `False`. Therefore, the corresponding implementation is as follows:

```python
class Threshold(nn.Cell):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def construct(self, inputs):
        cond = ops.gt(inputs, self.threshold)
        value = ops.fill(inputs.dtype, inputs.shape, self.value)
        return ops.select(cond, inputs, value)
```

Here `ops.gt`, `ops.fill`, and `ops.select` are used to implement judgment and replacement respectively. The following custom `Threshold` layer is implemented:

```python
m = Threshold(0.1, 20)
inputs = mindspore.Tensor([0.1, 0.2, 0.3], mindspore.float32)
m(inputs)
```

```text
Tensor(shape=[3], dtype=Float32, value= [ 2.00000000e+01,  2.00000003e-01,  3.00000012e-01])
```

It can be seen that `inputs[0] = threshold`, so it is replaced with `20`.

### Custom Cell Reverse

In special scenarios, we not only need to customize the forward logic of the neural network layer, but also want to manually control the computation of its reverse, which we can define through the `Cell.bprop` interface. The function will be used in scenarios such as new neural network structure design and backward propagation speed optimization. In the following, we take `Dropout2d` as an example to introduce custom Cell reverse.

```python
class Dropout2d(nn.Cell):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout2d = ops.Dropout2D(keep_prob)

    def construct(self, x):
        return self.dropout2d(x)

    def bprop(self, x, out, dout):
        _, mask = out
        dy, _ = dout
        if self.keep_prob != 0:
            dy = dy * (1 / self.keep_prob)
        dy = mask.astype(mindspore.float32) * dy
        return (dy.astype(x.dtype), )

dropout_2d = Dropout2d(0.8)
dropout_2d.bprop_debug = True
```

The `bprop` method has three separate input parameters:

- *x*: Forward input. When there are multiple forward inputs, the same number of inputs are required.
- *out*: Forward input.
- *dout*: When backward propagation is performed, the current Cell executes the previous reverse result.

Generally we need to calculate the reverse result according to the reverse derivative formula based on the forward output and the reverse result of the front layer, and return it. The reverse calculation of `Dropout2d` requires masking the reverse result of the front layer based on the `mask` matrix of the forward output, and then scaling according to `keep_prob`. The final implementation can get the correct calculation result.

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

For more descriptions of the HookBackward operator, refer to the [API documentation](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.HookBackward.html).

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

For more information about the `register_forward_pre_hook` function of the Cell object, refer to the [API documentation](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook).

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

For more information about the `register_forward_hook` function of the Cell object, please refer to the [API documentation](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook).

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
[[[[1.99999 1.99999]
   [1.99999 1.99999]]]]
-------------
 [[[[1.99999 1.99999]
   [1.99999 1.99999]]]]
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

For more information about the `register_backward_hook` function of the Cell object, please refer to the [API documentation](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook).
