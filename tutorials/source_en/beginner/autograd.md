# Automatic Differentiation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/autograd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Automatic differentiation can calculate a derivative value of a derivative function at a certain point, which is a generalization of backpropagation algorithms. The main problem solved by automatic differentiation is to decompose a complex mathematical operation into a series of simple basic operations. This function shields a large number of derivative details and processes from users, greatly reducing the threshold for using the framework.

MindSpore uses `ops.grad` and `ops.value_and_grad` to calculate the first-order derivative. `ops.grad` only returns gradient, while `ops.value_and_grad` returns the network forward calculation result and gradient. The `ops.value_and_grad` attributes are as follows:

- `fn`: the function or network to be derived.
- `grad_position`: specifies the index of the input position to be derived. If the index is int type, it means to derive for a single input; if tuple type, it means to derive for the position of the index within the tuple, where the index starts from 0; and if None, it means not to derive for the input. In this scenario, `weights` is non-None. Default: 0.
- `weights`: the network variables that need to return the gradients in the training network. Generally the network variables can be obtained by `weights = net.trainable_params()`. Default: None.
- `has_aux`: symbol for whether to return auxiliary arguments. If True, the number of `fn` outputs must be more than one, where only the first output of `fn` is involved in the derivation and the other output values will be returned directly. Default: False.

This chapter uses `ops.value_and_grad` in MindSpore to find first-order derivatives of the network.

## Finding Gradient of Network Weight

Since functional programming is suggested to use in MindSpore's automatic differentiation, the sample will be presented as functional programming.

```python
import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

# Define network
net = nn.Dense(10, 1)

# Define loss function
loss_fn = nn.MSELoss()

# Combine forward network and loss function
def forward(inputs, labels):
    logits = net(inputs)
    loss = loss_fn(logits, labels)
    return loss, logits
```

To find the first-order derivative of the weight parameter, you need to pass `weights` into `ops.value_and_grad`. There is no need to find derivative of the input, only to set `grad_position` to None.

Next, derive the network weights as follows:

```python
inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
labels = Tensor(np.random.randn(16, 1).astype(np.float32))
weights = net.trainable_params()

# has_aux is set to True, which means that only loss is used to derive, while logits is not.
grad_fn = ops.value_and_grad(forward, grad_position=None, weights=weights, has_aux=True)
(loss, logits), params_gradient = grad_fn(inputs, labels)

# Print result
print(logits.shape, len(weights), len(params_gradient))
```

```text
(16, 1) 2 2
```

## Stopping Calculating Gradients

 When the corresponding weight parameter declaration is defined, if some weights do not need to be derived, the attribute `requires_grad` needs to be set to `False` when defining the derivation network.

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = ms.Parameter(ms.Tensor(np.array([6], np.float32)), name='w')
        self.b = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='b', requires_grad=False)

    def construct(self, x):
        out = x * self.w + self.b
        return out

# Build derivation network
net = Net()
params = net.trainable_params()
x = ms.Tensor([5], dtype=ms.float32)
value, gradient = ops.value_and_grad(net, grad_position=None, weights=params)(x)

print(gradient)
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 5.00000000e+00]),)
```

Use `ops.stop_gradient` to stop calculating the gradient, for example:

```python
from mindspore.ops import stop_gradient

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = ms.Parameter(ms.Tensor(np.array([6], np.float32)), name='w')
        self.b = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='b')

    def construct(self, x):
        out = x * self.w + self.b
        # Stop updating gradient, and the out does not contribute to gradient calculation
        out = stop_gradient(out)
        return out

net = Net()
params = net.trainable_params()
x = ms.Tensor([100], dtype=ms.float32)
value, output = ops.value_and_grad(net, grad_position=None, weights=params)(x)

print(f"wgrad: {output[0]}\nbgrad: {output[1]}")
```

```text
wgrad: [0.]
bgrad: [0.]
```
