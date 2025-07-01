[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/autograd.md)

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || **Autograd** || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/master/beginner/mixed_precision.html)

# Automatic Differentiation

The training of the neural network mainly uses the back propagation algorithm. Model predictions (logits) and the correct labels are fed into the loss function to obtain the loss, and then the back propagation calculation is performed to obtain the gradients, which are finally updated to the model parameters. Automatic differentiation is able to calculate the value of the derivative of a derivable function at a point and is a generalization of the backpropagation algorithm. The main problem solved by automatic differentiation is to decompose a complex mathematical operation into a series of simple basic operations. The function shields the user from a large number of derivative details and processes, which greatly reduces the threshold of using the framework.

MindSpore uses the design philosophy of functional auto-differentiation to provide auto-differentiation interfaces [mindspore.grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html) and [mindspore.value_and_grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.value_and_grad.html) that are closer to the mathematical semantics. We introduce it below by using a simple single-level linear transform model.

```python
import numpy as np
import mindspore
from mindspore import ops, nn, Tensor, Parameter
```

## Functions and Computing Graphs

Computing graphs are a way to represent mathematical functions in a graph-theoretic language and a unified way to represent neural network models in a deep learning framework. We will construct computing functions and neural networks based on the following computing graphs.

![compute-graph](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/comp-graph.png)

In this model, $x$ is the input, $y$ is the correct value, and $w$ and $b$ are the parameters we need to optimize.

```python
np.random.seed(42)
x = ops.ones(5, mindspore.float32)  # input tensor
y = ops.zeros(3, mindspore.float32)  # expected output
w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias
```

We construct the computing function based on the computing process described by the computing graphs.

[binary_cross_entropy_with_logits](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.binary_cross_entropy_with_logits.html) is a loss function,
which computes binary cross entropy between the logits and the label.

```python
def function(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss
```

Execute the computing functions to get the calculated loss value.

```python
loss = function(x, y, w, b)
print(loss)
```

```text
1.342315
```

## Differential Functions and Gradient Computing

In order to optimize the model parameters, find the derivatives of the parameters with respect to loss: $\frac{\partial \operatorname{loss}}{\partial w}$ and $\frac{\partial \operatorname{loss}}{\partial b}$. At this point we call the `ops. grad` function to get the differential function of `function`.

Two input parameters of `grad` function are used here:

- `fn`: the function to be derived.
- `grad_position`: specifies the index of the input position for the derivative.

Since we derive $w$ and $b$, we configure their positions `(2, 3)` corresponding to the `function` input parameter.

> Using `grad` to obtain a differential function is a functional transform, i.e. the input is a function and the output is also a function.

```python
grad_fn = mindspore.grad(function, (2, 3))
```

The gradients corresponding to $w$ and $b$ are obtained by executing the differentiation function.

```python
grads = grad_fn(x, y, w, b)
print(grads)
```

```text
(Tensor(shape=[5, 3], dtype=Float32, value=
 [[ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02]]), Tensor(shape=[3], dtype=Float32, value= [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02]))
```

## Stop Gradient

Generally, the derivative is obtained by taking the derivative of loss with respect to the parameter, so that the output of function is only one term of loss. When we want the function to output more than one term, the differential function will find the derivative of all output terms with respect to the parameter. In this case, if you want to truncate the gradient of an output term or eliminate the effect of a Tensor on the gradient, you need to use Stop Gradient operation.

Here we change `function` to `function_with_logits` that outputs both loss and z to obtain the differentiation function and execute it.

```python
def function_with_logits(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, z
```

```python
grad_fn = mindspore.grad(function_with_logits, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)
```

```text
(Tensor(shape=[5, 3], dtype=Float32, value=
 [[ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00],
  [ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00],
  [ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00],
  [ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00],
  [ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00]]), Tensor(shape=[3], dtype=Float32, value= [ 1.06568694e+00,  1.05373347e+00,  1.30146706e+00]))
```

You can see that the gradient values corresponding to $w$ and $b$ have changed. At this point, if you want to block out the effect of z on the gradient, i.e., still only find the derivative of the parameter with respect to loss, you can use the [mindspore.ops.stop_gradient](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.stop_gradient.html) interface to truncate the gradient here. We add the `function` implementation to `stop_gradient` and execute it.

```python
def function_stop_gradient(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, ops.stop_gradient(z)
```

```python
grad_fn = mindspore.grad(function_stop_gradient, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)
```

```text
(Tensor(shape=[5, 3], dtype=Float32, value=
 [[ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00],
  [ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00],
  [ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00],
  [ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00],
  [ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00]]), Tensor(shape=[3], dtype=Float32, value= [ 1.32618928e+00,  1.01589143e+00,  1.04216456e+00]))
```

It can be seen that the gradient values corresponding to $w$ and $b$ are the same as the gradient values found by the initial `function`.

## Auxiliary data

Auxiliary data is other outputs of the function in addition to the first output items. Usually we will set loss of the function as the first output, the other output is the auxiliary data.

`grad` and `value_and_grad` provide `has_aux` parameter. When it is set to `True`, it can automatically implement the function of manually adding `stop_gradient` in the previous section, satisfying the effect of returning auxiliary data without affecting the gradient calculation.

The following still uses `function_with_logits`, configures `has_aux=True`, and executes it.

```python
grad_fn = mindspore.grad(function_with_logits, (2, 3), has_aux=True)
```

```python
grads, (z,) = grad_fn(x, y, w, b)
print(grads, z)
```

```text
(Tensor(shape=[5, 3], dtype=Float32, value=
 [[ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02]]), Tensor(shape=[3], dtype=Float32, value= [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02])) [ 3.8211915 -2.994512  -1.932323 ]
```

## Calculating Neural Network Gradient

The previous section introduces MindSpore functional auto-differentiation based mainly on the functions corresponding to the computing graph, but neural network construction is inherited from the object-oriented programming paradigm of `nn.Cell`. Next, we construct the same neural network by `Cell` and use functional automatic differentiation to implement backpropagation.

First we inherit `nn.Cell` to construct a single-layer linear transform neural network. Here we directly use $w$, $b$ from the previous section as model parameters, wrapped with `mindspore.Parameter` as internal properties, and implement the same Tensor operations within `construct`.

```python
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = w
        self.b = b

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z
```

Next we instantiate the model and the loss function.

```python
# Instantiate model
model = Network()
# Instantiate loss function
loss_fn = nn.BCEWithLogitsLoss()
```

Once completed, the calls to the neural network and loss function need to be encapsulated into a forward computing function due to the need to use functional automatic differentiation.

```python
# Define forward function
def forward_fn(x, y):
    z = model(x)
    loss = loss_fn(z, y)
    return loss
```

Once completed, we use the `value_and_grad` interface to obtain the differentiation function for computing the gradient.

Since Cell is used to encapsulate the neural network model and the model parameters are internal properties of Cell, we do not need to use `grad_position` to specify the derivation of the function inputs at this point, so we configure it as `None`. When derive the model parameters, we use the `weights` parameter and use the `model.trainable_params()` method to retrieve the parameters from the Cell that can be derived.

```python
grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())
```

```python
loss, grads = grad_fn(x, y)
print(grads)
```

```text
(Tensor(shape=[5, 3], dtype=Float32, value=
 [[ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02],
  [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02]]), Tensor(shape=[3], dtype=Float32, value= [ 3.26189250e-01,  1.58914644e-02,  4.21645455e-02]))
```

Executing the differentiation function, and we can see that the gradient value is the same as the gradient value obtained from the previous `function`.
