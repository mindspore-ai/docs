# Gradient Derivation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/model_development/gradient.md)

## Automatic Differentiation

Both MindSpore and PyTorch provide the automatic differentiation function. After the forward network is defined, automatic backward propagation and gradient update can be implemented through simple interface invoking. However, it should be noted that MindSpore and PyTorch use different logic to build backward graphs. This difference also brings differences in API design.

<table>
<tr>
<td style="text-align:center"> PyTorch Automatic Differentiation </td> <td style="text-align:center"> MindSpore Automatic Differentiation </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# Note: The feedback of PyTorch is cumulative,
# and after updating, the optimizer needs to be cleared.

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
x = x * 2
y = x - 1
y.backward(x)

```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# ms.grad: forward graph as input, backward graph as output.
import mindspore as ms
from mindspore import nn
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net

    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
        return gradient_function(x, y)
```

</pre>
</td>
</tr>
</table>

### Principle Comparison

#### PyTorch Automatic Differentiation

As we know, PyTorch is an automatic differentiation based on computation path tracing. After a network structure is defined, no backward graph is created. Instead, during the execution of the forward graph, `Variable` or `Parameter` records the backward function corresponding to each forward computation and generates a dynamic computational graph, it is used for subsequent gradient calculation. When `backward` is called at the final output, the chaining rule is applied to calculate the gradient from the root node to the leaf node. The nodes stored in the dynamic computational graph of PyTorch are actually `Function` objects. Each time an operation is performed on `Tensor`, a `Function` object is generated, which records necessary information in backward propagation. During backward propagation, the `autograd` engine calculates gradients in backward order by using the `backward` of the `Function`. You can view this point through the hidden attribute of the `Tensor`.

#### MindSpore Automatic Differentiation

In graph mode, MindSpore's automatic differentiation is based on the graph structure. Different from PyTorch, MindSpore does not record any information during forward computation and only executes the normal computation process (similar to PyTorch in PyNative mode). Then the question comes. If the entire forward computation is complete and MindSpore does not record any information, how does MindSpore know how backward propagation is performed?

When MindSpore performs automatic differentiation, the forward graph structure needs to be transferred. The automatic differentiation process is to obtain backward propagation information by analyzing the forward graph. The automatic differentiation result is irrelevant to the specific value in the forward computation and is related only to the forward graph structure. Through the automatic differentiation of the forward graph, the backward propagation process is obtained. The backward propagation process is expressed through a graph structure, that is, the backward graph. The backward graph is added after the user-defined forward graph to form a final computational graph. However, the backward graph and backward operators added later are not aware of and cannot be manually added. They can only be automatically added through the interface provided by MindSpore. In this way, errors are avoided during backward graph build.

Finally, not only the forward graph is executed, but also the graph structure contains both the forward operator and the backward operator added by MindSpore. That is, MindSpore adds an invisible `Cell` after the defined forward graph, the `Cell` is a backward operator derived from the forward graph.

The interface that helps us build the backward graph is [grad](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.grad.html).

After that, for any group of data you enter, it can calculate not only the positive output, but also the gradient of ownership weight. Because the graph structure is fixed and does not save intermediate variables, the graph structure can be invoked repeatedly.

Similarly, when we add an optimizer structure to the network, the optimizer also adds optimizer-related operators. That is, we add optimizer operators that are not perceived to the computational graph. Finally, the computational graph is built.

In MindSpore, most operations are finally converted into real operator operations and finally added to the computational graph. Therefore, the number of operators actually executed in the computational graph is far greater than the number of operators defined at the beginning.

MindSpore provides the [TrainOneStepCell](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.TrainOneStepCell.html) and [TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html) APIs to package the entire training process. If other operations, such as gradient cropping, specification, and intermediate variable return, are performed in addition to the common training process, you need to customize the training cell. For details, see [Inference and Training Process](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/training_and_evaluation.html).

### Interface Comparison

#### torch.autograd.backward

[torch.autograd.backward](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html). For a scalar, calling its backward method automatically computes the gradient values of the leaf nodes according to the chaining law. For vectors and matrices, you need to define grad_tensor to compute the gradient of the matrix.
Typically after calling backward once, PyTorch automatically destroys the computation graph, so to call backward repeatedly on a variable, you need to set the return_graph parameter to True.
If you need to compute higher-order gradients, you need to set create_graph to True.
The two expressions z.backward() and torch.autograd.backward(z) are equivalent.

```python
import torch
print("=== tensor.backward ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
print("x.grad before backward", x.grad)
print("y.grad before backward", y.grad)
z.backward()
print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)

print("=== torch.autograd.backward ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
torch.autograd.backward(z)
print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)
```

Outputs:

```text
=== tensor.backward ===
x.grad before backward None
y.grad before backward None
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
=== torch.autograd.backward ===
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
```

It can be seen that before calling the backward function, x.grad and y.grad functions are empty. And after the backward calculation, x.grad and y.grad represent the values after the derivative calculation, respectively.

This interface is implemented in MindSpore using mindspore.grad. The above PyTorch use case can be transformed into:

```python
import mindspore
print("=== mindspore.grad ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y
out = mindspore.grad(net, grad_position=0)(x, y)
print("out", out)
out1 = mindspore.grad(net, grad_position=1)(x, y)
print("out1", out1)
```

Outputs:

```text
=== mindspore.grad ===
out 2.0
out1 1.0
```

If the above net has more than one output, you need to pay attention to the effect of multiple outputs of the network on finding the gradient.

```python
import mindspore
print("=== mindspore.grad multiple outputs ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y, x
out = mindspore.grad(net, grad_position=0)(x, y)
print("out", out)
out1 = mindspore.grad(net, grad_position=1)(x, y)
print("out1", out)
```

Outputs:

```text
=== mindspore.grad multiple outputs ===
out 3.0
out1 3.0
```

PyTorch does not support such expressions:

```python
import torch
print("=== torch.autograd.backward does not support multiple outputs ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
torch.autograd.backward(z)
print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)
```

Outputs:

```text
=== torch.autograd.backward does not support multiple outputs ===
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
```

Therefore, to find the gradient of only the first output in MindSpore, you need to use the has_aux parameter in MindSpore.

```python
import mindspore
print("=== mindspore.grad has_aux ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y, x
grad_fcn = mindspore.grad(net, grad_position=0, has_aux=True)
out, _ = grad_fcn(x, y)
print("out", out)
grad_fcn1 = mindspore.grad(net, grad_position=1, has_aux=True)
out, _ = grad_fcn1(x, y)
print("out", out)
```

Outputs:

```text
=== mindspore.grad has_aux ===
out 2.0
out 1.0
```

#### torch.autograd.grad

[torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html). This interface is basically the same as torch.autograd.backward. The difference between the two is that the former modifies the grad attribute of each Tensor directly, while the latter returns a list of gradient values for the parameters. So when migrating to MindSpore, you can also refer to the above use case.

```python
import torch
print("=== torch.autograd.grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
out = torch.autograd.grad(z, x)
out1 = torch.autograd.grad(z, y)
print("out", out)
print("out1", out1)
```

Outputs:

```text
=== torch.autograd.grad ===
out (tensor(2.),)
out1 (tensor(1.),)
```

#### torch.no_grad

In PyTorch, by default, information required for backward propagation is recorded when forward computation is performed. In the inference phase or in a network where backward propagation is not required, this operation is redundant and time-consuming. Therefore, PyTorch provides `torch.no_grad` to cancel this process.

MindSpore constructs a backward graph based on the forward graph structure only when `grad` is invoked. No information is recorded during forward execution. Therefore, MindSpore does not need this interface. It can be understood that forward calculation of MindSpore is performed in `torch.no_grad` mode.

```python
import torch
print("=== torch.no_grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
print("z.requires_grad", z.requires_grad)
with torch.no_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
```

Outputs:

```text
=== torch.no_grad ===
z.requires_grad True
z.requires_grad False
```

#### torch.enable_grad

If PyTorch enables `torch.no_grad` to disable gradient computation, you can use this interface to enable it.

MindSpore builds the backward graph based on the forward graph structure only when `grad` is called, and no information is logged during forward execution, so MindSpore doesn't need this interface, and it can be understood that MindSpore backward computations are performed with `torch.enable_grad`.

```python
import torch
print("=== torch.enable_grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
with torch.enable_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
```

Outputs:

```text
=== torch.enable_grad ===
z.requires_grad False
z.requires_grad True
```

#### retain_graph

PyTorch is function-based automatic differentiation. Therefore, by default, the recorded information is automatically cleared after each backward propagation is performed for the next iteration. As a result, when we want to reuse the backward graph and gradient information, the information fails to be obtained because it has been deleted. Therefore, PyTorch provides `backward(retain_graph=True)` to proactively retain the information.

MindSpore does not require this function. MindSpore is an automatic differentiation based on the computational graph. The backward graph information is permanently recorded in the computational graph after `grad` is invoked. You only need to invoke the computational graph again to obtain the gradient information.

## Automatic Differentiation Interfaces

After the forward network is constructed, MindSpore provides an interface to [automatic differentiation](https://mindspore.cn/tutorials/en/r2.3/beginner/autograd.html) to calculate the gradient results of the model.
In the tutorial of [automatic derivation](https://mindspore.cn/tutorials/en/r2.3/advanced/derivation.html), some descriptions of various gradient calculation scenarios are given.

### mindspore.grad

There are four configurable parameters in [mindspore.grad](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.grad.html):

- fn (Union[Cell, Function]) - The function or network (Cell) to be derived.

- grad_position (Union[NoneType, int, tuple[int]]) - Specifies the index of the input position for the derivative. Default value: 0.

- weights (Union[ParameterTuple, Parameter, list[Parameter]]) - The network parameter that needs to return the gradient in the training network. Default value: None.

- has_aux (bool) - Mark for whether to return the auxiliary parameters. If True, the number of fn outputs must be more than one, where only the first output of fn is involved in the derivation and the other output values will be returned directly. Default value: False.

where `grad_position` and `weights` together determine which values of the gradient are to be output, and has_aux configures whether to find the gradient on the first input or on all outputs when there are multiple outputs.

| grad_position | weights | output |
| ------------- | ------- | ------ |
| 0         | None   | Gradient of the first input |
| 1         | None   | Gradient of the second input |
| (0, 1)      | None   | (Gradient of the first input, gradient of the second input) |
| None       | weights | (Gradient of weights) |
| 0         | weights | (Gradient of the first input), (Gradient of weights) |
| (0, 1)      | weights | (Gradient of the first input, Gradient of the second input), (Gradient of weights) |
| None       | None   | Report an error  |

Run an actual example to see exactly how it works.

First, a network with parameters is constructed, which has two outputs loss and logits, where loss is the output we use to find the gradient.

```python
import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(Net, self).__init__()
        self.fc = nn.Dense(in_channel, out_channel, has_bias=False)
        self.loss = nn.MSELoss()

    def construct(self, x, y):
        logits = self.fc(x).squeeze()
        loss = self.loss(logits, y)
        return loss, logits


net = Net(3, 1)
net.fc.weight.set_data(ms.Tensor([[2, 3, 4]], ms.float32))   # Set a fixed value for fully connected weight

print("=== weight ===")
for param in net.trainable_params():
    print("name:", param.name, "data:", param.data.asnumpy())
x = ms.Tensor([[1, 2, 3]], ms.float32)
y = ms.Tensor(19, ms.float32)

loss, logits = net(x, y)
print("=== output ===")
print(loss, logits)
```

Outputs:

```text
=== weight ===
name: fc.weight data: [[2. 3. 4.]]
=== output ===
1.0 20.0
```

```python
# Find the gradient for the first input

print("=== grads 1 ===")
grad_func = ms.grad(net, grad_position=0, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

Outputs:

```text
=== grads 1 ===
grad [[4. 6. 8.]]
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Find the gradient for the second input

print("=== grads 2 ===")
grad_func = ms.grad(net, grad_position=1, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

Outputs:

```text
=== grads 2 ===
grad -2.0
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Finding the gradient for multiple inputs

print("=== grads 3 ===")
grad_func = ms.grad(net, grad_position=(0, 1), weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

Outputs:

```text
=== grads 3 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Find the gradient for weights

print("=== grads 4 ===")
grad_func = ms.grad(net, grad_position=None, weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logits", logit)
```

Outputs:

```text
=== grads 4 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),)
logits (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Find the gradient for the first input and weights

print("=== grads 5 ===")
grad_func = ms.grad(net, grad_position=0, weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

Outputs:

```text
=== grads 5 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Find the gradient for multiple inputs and weights

print("=== grads 6 ===")
grad_func = ms.grad(net, grad_position=(0, 1), weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

Outputs:

```text
=== grads 6 ===
grad ((Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2)), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Scenario with has_aux=False

print("=== grads 7 ===")
grad_func = ms.grad(net, grad_position=0, weights=None, has_aux=False)
grad = grad_func(x, y)  # Only one output
print("grad", grad)
```

Outputs:

```text
=== grads 7 ===
grad [[ 6.  9. 12.]]
```

The `has_aux=False` scenario is actually equivalent to summing two outputs as the output of finding the gradient:

```python
class Net2(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = nn.Dense(in_channel, out_channel, has_bias=False)
        self.loss = nn.MSELoss()

    def construct(self, x, y):
        logits = self.fc(x).squeeze()
        loss = self.loss(logits, y)
        return loss + logits

net2 = Net2(3, 1)
net2.fc.weight.set_data(ms.Tensor([[2, 3, 4]], ms.float32))   # Set a fixed value for fully connected weight
grads = ms.grad(net2, grad_position=0, weights=None, has_aux=False)
grad = grads(x, y)  # Only one output
print("grad", grad)
```

Outputs:

```text
grad [[ 6.  9. 12.]]
```

```python
# grad_position=None, weights=None

print("=== grads 8 ===")
grad_func = ms.grad(net, grad_position=None, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)

# === grads 8 ===
# ValueError: `grad_position` and `weight` can not be None at the same time.
```

### mindspore.value_and_grad

The parameters of the interface [mindspore.value_and_grad](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.value_and_grad.html) is the same as that of the above grad, except that this interface calculates the forward result and gradient of the network at once.

| grad_position | weights | output |
| ------------- | ------- | ------ |
| 0         | None   | (Output of the network, gradient of the first input) |
| 1         | None   | (Output of the network, gradient of the second input) |
| (0, 1)      | None   | (Output of the network, (Gradient of the first input, gradient of the second input)) |
| None       | weights | (Output of the network, (gradient of the weights)) |
| 0         | weights | (Output of the network, ((Gradient of the first input), (gradient of the weights))) |
| (0, 1)      | weights | (Output of the network, ((Gradient of the first input, gradient of the second input), (gradient of the weights))) |
| None       | None   | Report an error  |

```python
print("=== value and grad ===")
value_and_grad_func = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params(), has_aux=True)
value, grad = value_and_grad_func(x, y)
print("value", value)
print("grad", grad)
```

Outputs:

```text
=== value and grad ===
value (Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 20))
grad ((Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2)), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
```

### mindspore.ops.GradOperation

[mindspore.ops.GradOperation](https://mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.GradOperation.html), a higher-order function that generates a gradient function for the input function.

The gradient function generated by the GradOperation higher-order function can be customized by the construction parameters.

This function is similar to the function of grad, and it is not recommended in the current version. Please refer to the description in the API for details.

## loss scale

Since the gradient overflow may be encountered in the process of finding the gradient in the mixed accuracy scenario, we generally use the loss scale to accompany the gradient derivation.

> On Ascend, because operators such as Conv, Sort, and TopK can only be float16, and MatMul is preferably float16 due to performance issues, it is recommended that loss scale operations be used as standard for network training. [List of operators on Ascend only support float16][https://www.mindspore.cn/docs/en/r2.3/migration_guide/debug_and_tune.html#4-training-accuracy].
>
> The overflow can obtain overflow operator information via MindSpore Insight [debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger.html) or [dump data](https://mindspore.cn/tutorials/experts/en/r2.3/debug/dump.html).
>
> General overflow manifests itself as loss Nan/INF, loss suddenly becomes large, etc.

```python
from mindspore.amp import StaticLossScaler, all_finite

loss_scale = StaticLossScaler(1024.)  #  Static lossscale

def forward_fn(x, y):
    loss, logits = net(x, y)
    print("loss", loss)
    loss = loss_scale.scale(loss)
    return loss, logits

value_and_grad_func = ms.value_and_grad(forward_fn, grad_position=None, weights=net.trainable_params(), has_aux=True)
(loss, logits), grad = value_and_grad_func(x, y)
print("=== loss scale ===")
print("loss", loss)
print("grad", grad)
print("=== unscale ===")
loss = loss_scale.unscale(loss)
grad = loss_scale.unscale(grad)
print("loss", loss)
print("grad", grad)

# Check whether there is an overflow, and return True if there is no overflow
state = all_finite(grad)
print(state)
```

Outputs:

```text
loss 1.0
=== loss scale ===
loss 1024.0
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.04800000e+003, 4.09600000e+003, 6.14400000e+003]]),)
=== unscale ===
loss 1.0
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),)
True
```

The principle of loss scale is very simple. By multiplying a relatively large value for loss, through the chain conduction of the gradient, a relatively large value is multiplied on the link of calculating the gradient, to prevent accuracy problems from occurring when the gradient is too small during the back propagation.

After calculating the gradient, you need to divide the loss and gradient back to the original value to ensure that the whole calculation process is correct.

Finally, you generally need to use all_finite to determine if there is an overflow, and if there is no overflow you can use the optimizer to update the parameters.

## Gradient Cropping

When the training process encountered gradient explosion or particularly large gradient, and training instability, you can consider adding gradient cropping. Here is an example of using global_norm for gradient cropping scenarios:

```python
from mindspore import ops

grad = ops.clip_by_global_norm(grad)
```

## Gradient Accumulation

Gradient accumulation is a way that data samples of a kind of training neural network is split into several small Batches by Batch, and then calculated in order to solve the OOM (Out Of Memory) problem that due to the lack of memory, resulting in too large Batch size, the neural network can not be trained or the network model is too large to load.

For detailed, refer to [Gradient Accumulation](https://www.mindspore.cn/tutorials/experts/en/r2.3/optimize/gradient_accumulation.html).
