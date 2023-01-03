# Gradient Derivation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/model_development/gradient.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Automatic Differentiation Interfaces

After the forward network is constructed, MindSpore provides an interface to [automatic differentiation](https://mindspore.cn/tutorials/en/master/beginner/autograd.html) to calculate the gradient results of the model.
In the tutorial of [automatic derivation](https://mindspore.cn/tutorials/en/master/advanced/derivation.html), some descriptions of various gradient calculation scenarios are given.

There are three MindSpore interfaces for finding gradients currently.

### mindspore.grad

There are four configurable parameters in [mindspore.grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html):

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

```text
=== grads 4 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),)
logits (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# Find the gradient for the first output and weights

print("=== grads 5 ===")
grad_func = ms.grad(net, grad_position=0, weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

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

The parameters of the interface [mindspore.value_and_grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.value_and_grad.html) is the same as that of the above grad, except that this interface calculates the forward result and gradient of the network at once.

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

```text
=== value and grad ===
value (Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 20))
grad ((Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2)), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
```

### mindspore.ops.GradOperation

[mindspore.ops.GradOperation](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.GradOperation.html), a higher-order function that generates a gradient function for the input function.

The gradient function generated by the GradOperation higher-order function can be customized by the construction parameters.

This function is similar to the function of grad, and it is not recommended in the current version. Please refer to the description in the API for details.

## loss scale

Since the gradient overflow may be encountered in the process of finding the gradient in the mixed accuracy scenario, we generally use the loss scale to accompany the gradient derivation.

> On Ascend, because operators such as Conv, Sort, and TopK can only be float16, and MatMul is preferably float16 due to performance issues, it is recommended that loss scale operations be used as standard for network training. [List of operators on Ascend only support float16][https://www.mindspore.cn/docs/en/master/migration_guide/debug_and_tune.html#training-accuracy].
>
> The overflow can obtain overflow operator information via MindInsight [debugger](https://www.mindspore.cn/mindinsight/docs/en/master/debugger.html) or [dump data](https://mindspore.cn/tutorials/experts/en/master/debug/dump.html).
>
> General overflow manifests itself as loss Nan/INF, loss suddenly becomes large, etc.

```python
from mindspore.amp import StaticLossScaler, all_finite

loss_scale = StaticLossScaler(1024.)  # 静态lossscale

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

For detailed, refer to [Gradient Accumulation](https://www.mindspore.cn/tutorials/experts/en/master/optimize/gradient_accumulation.html).
