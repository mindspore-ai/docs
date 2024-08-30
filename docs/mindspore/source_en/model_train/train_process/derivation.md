# Advanced Automatic Differentiation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/train_process/derivation.md)

The `grad` and `value_and_grad` provided by the `mindspore.ops` module generate the gradients of the network model. `grad` computes the network gradient, and `value_and_grad` computes both the forward output and the gradient of the network. This article focuses on how to use the main functions of the `grad`, including first-order and second-order derivations, derivation of the input or network weights separately, returning auxiliary variables, and stopping calculating the gradient.

> For more information about the derivative interface, please refer to the [API documentation](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html).

## First-order Derivation

Method: `mindspore.grad`. The parameter usage is as follows:

- `fn`: the function or network to be derived.
- `grad_position`: specifies the index of the input position to be derived. If the index is int type, it means to derive for a single input; if tuple type, it means to derive for the position of the index within the tuple, where the index starts from 0; and if None, it means not to derive for the input. In this scenario, `weights` is non-None. Default: 0.
- `weights`: the network variables that need to return the gradients in the training network. Generally the network variables can be obtained by `weights = net.trainable_params()`. Default: None.
- `has_aux`: symbol for whether to return auxiliary arguments. If True, the number of `fn` outputs must be more than one, where only the first output of `fn` is involved in the derivation and the other output values will be returned directly. Default: False.

The following is a brief introduction to the use of the `grad` by first constructing a customized network model `Net` and then performing a first-order derivative on it:

$$f(x, y)=(x * z) * y \tag{1}$$

First, define the network model `Net`, input `x`, and input `y`.

```python
import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

# Define the inputs x and y.
x = Tensor([3.0], dtype=ms.float32)
y = Tensor([5.0], dtype=ms.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        out = x * x * y * self.z
        return out
```

### Computing the First-order Derivative for Input

To derive the inputs `x` and `y`, set `grad_position` to (0, 1):

$$\frac{\partial f}{\partial x}=2 * x * y * z \tag{2}$$

$$\frac{\partial f}{\partial y}=x * x * z \tag{3}$$

```python
net = Net()
grad_fn = ms.grad(net, grad_position=(0, 1))
gradients = grad_fn(x, y)
print(gradients)
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 3.00000000e+01]), Tensor(shape=[1], dtype=Float32, value= [ 9.00000000e+00]))
```

### Computing the Derivative for Weight

Derive for the weight `z`, where it is not necessary to derive for the inputs, and set `grad_position` to None:

$$\frac{\partial f}{\partial z}=x * x * y \tag{4}$$

```python
params = ms.ParameterTuple(net.trainable_params())

output = ms.grad(net, grad_position=None, weights=params)(x, y)
print(output)
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 4.50000000e+01]),)
```

### Returning Auxiliary Variables

Simultaneous derivation for the inputs and weights, where only the first output is involved in the derivation, with the following sample code:

```python
net = nn.Dense(10, 1)
loss_fn = nn.MSELoss()


def forward(inputs, labels):
    logits = net(inputs)
    loss = loss_fn(logits, labels)
    return loss, logits


inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
labels = Tensor(np.random.randn(16, 1).astype(np.float32))
weights = net.trainable_params()

# Aux value does not contribute to the gradient.
grad_fn = ms.grad(forward, grad_position=0, weights=None, has_aux=True)
inputs_gradient, (aux_logits,) = grad_fn(inputs, labels)
print(len(inputs_gradient), aux_logits.shape)
```

```text
16, (16, 1)
```

### Stopping Gradient Computation

You can use `stop_gradient` to stop computing the gradient of a specified operator to eliminate the impact of the operator on the gradient.

Based on the matrix multiplication network model used for the first-order derivation, add an operator `out2` and disable the gradient computation to obtain the customized network `Net2`. Then, check the derivation result of the input.

The sample code is as follows:

```python
class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        out1 = x * y
        out2 = x * y
        out2 = ops.stop_gradient(out2) # Stop computing the gradient of the out2 operator.
        out = out1 + out2
        return out


net = Net()
grad_fn = ms.grad(net)
output = grad_fn(x, y)
print(output)
```

```text
[5.0]
```

According to the preceding information, `stop_gradient` is set for `out2`. Therefore, `out2` does not contribute to gradient computation. The output result is the same as that when `out2` is not added.

Delete `out2 = stop_gradient(out2)` and check the output result. An example of the code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        out1 = x * y
        out2 = x * y
        # out2 = stop_gradient(out2)
        out = out1 + out2
        return out


net = Net()
grad_fn = ms.grad(net)
output = grad_fn(x, y)
print(output)
```

```text
[10.0]
```

According to the printed result, after the gradient of the `out2` operator is computed, the gradients generated by the `out2` and `out1` operators are the same. Therefore, the value of each item in the result is twice the original value (accuracy error exists).

## High-order Derivation

High-order differentiation is used in domains such as AI-supported scientific computing and second-order optimization. For example, in the molecular dynamics simulation, when the potential energy is trained using the neural network, the derivative of the neural network output to the input needs to be computed in the loss function, and then the second-order cross derivative of the loss function to the input and the weight exists in backward propagation.

In addition, the second-order derivative of the output to the input exists in a differential equation solved by AI (such as PINNs). Another example is that in order to enable the neural network to converge quickly in the second-order optimization, the second-order derivative of the loss function to the weight needs to be computed using the Newton method.

MindSpore can support high-order derivatives by computing derivatives for multiple times. The following uses several examples to describe how to compute derivatives.

### Single-input Single-output High-order Derivative

For example, the formula of the Sin operator is as follows:

$$f(x) = sin(x) \tag{1}$$

The first derivative is:

$$f'(x) = cos(x) \tag{2}$$

The second derivative is:

$$f''(x) = cos'(x) = -sin(x) \tag{3}$$

The second derivative (-Sin) is implemented as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""

    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out

x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net)
secondgrad = ms.grad(firstgrad)
output = secondgrad(x_train)

# Print the result.
result = np.around(output.asnumpy(), decimals=2)
print(result)
```

```text
[-0.]
```

The preceding print result shows that the value of $-sin(3.1415926)$ is close to $0$.

### Single-input Multi-output High-order Derivative

Compute the derivation of the following formula:

$$f(x) = (f_1(x), f_2(x)) \tag{1}$$

Where:

$$f_1(x) = sin(x) \tag{2}$$

$$f_2(x) = cos(x) \tag{3}$$

MindSpore uses the reverse-mode automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input is computed. Therefore, the first derivative is:

$$f'(x) = cos(x)  -sin(x) \tag{4}$$

The second derivative is:

$$f''(x) = -sin(x) - cos(x) \tag{5}$$

```python
import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x):
        out1 = self.sin(x)
        out2 = self.cos(x)
        return out1, out2

x_train = Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net)
secondgrad = ms.grad(firstgrad)
output = secondgrad(x_train)

# Print the result.
result = np.around(output.asnumpy(), decimals=2)
print(result)
```

```text
[1.]
```

The preceding print result shows that the value of $-sin(3.1415926) - cos(3.1415926)$ is close to $1$.

### Multiple-Input Multiple-Output High-Order Derivative

Compute the derivation of the following formula:

$$f(x, y) = (f_1(x, y), f_2(x, y)) \tag{1}$$

Where:

$$f_1(x, y) = sin(x) - cos(y)  \tag{2}$$

$$f_2(x, y) = cos(x) - sin(y)  \tag{3}$$

MindSpore uses the reverse-mode automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input is computed.

Sum:

$$\sum{output} = sin(x) + cos(x) - sin(y) - cos(y) \tag{4}$$

The first derivative of output sum with respect to input $x$ is:

$$\dfrac{\mathrm{d}\sum{output}}{\mathrm{d}x} = cos(x) - sin(x) \tag{5}$$

The second derivative of output sum with respect to input $x$ is:

$$\dfrac{\mathrm{d}\sum{output}^{2}}{\mathrm{d}^{2}x} = -sin(x) - cos(x) \tag{6}$$

The first derivative of output sum with respect to input $y$ is:

$$\dfrac{\mathrm{d}\sum{output}}{\mathrm{d}y} = -cos(y) + sin(y) \tag{7}$$

The second derivative of output sum with respect to input $y$ is:

$$\dfrac{\mathrm{d}\sum{output}^{2}}{\mathrm{d}^{2}y} = sin(y) + cos(y) \tag{8}$$

```python
import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        out1 = self.sin(x) - self.cos(y)
        out2 = self.cos(x) - self.sin(y)
        return out1, out2

x_train = Tensor(np.array([3.1415926]), dtype=ms.float32)
y_train = Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net, grad_position=(0, 1))
secondgrad = ms.grad(firstgrad, grad_position=(0, 1))
output = secondgrad(x_train, y_train)

# Print the result.
print(np.around(output[0].asnumpy(), decimals=2))
print(np.around(output[1].asnumpy(), decimals=2))
```

```text
[1.]
[-1.]
```

According to the preceding result, the value of the second derivative $-sin(3.1415926) - cos(3.1415926)$ of the output to the input $x$ is close to $1$, and the value of the second derivative $sin(3.1415926) + cos(3.1415926)$ of the output to the input $y$ is close to $-1$.

> The accuracy may vary depending on the computing platform. Therefore, the execution results of the code in this section vary slightly on different platforms.

## Computing Jacobian and Hessian Matrices Using Functional Transformations

### Jacobian Matrices

Before describing the methods provided by MindSpore to compute Jacobian matrices, an introduction to Jacobian matrices is first given.

We first define a mapping $\textbf{f}$:

$$R^{n} \longrightarrow R^{m}$$

We use the notation $\longmapsto$ here to denote the mapping between the elements of the set, and bold all the symbols representing vectors:

$$\textbf{x} \longmapsto \textbf{f}(\textbf{x})$$

where $\textbf{x} = (x_{1}, x_{2},\dots, x_{n})$，$\textbf{f(x)} = (f_{1}(\textbf{x}), f_{2}(\textbf{x}),\dots,f_{m}(\textbf{x}))$.

We denote the set consisting of all mappings from $R^{n}$ to $R^{m}$ as $F_{n}^{m}$.
Here, we refer to a mapping from one function (mapping) set to another function (mapping) set as an operation. It is easy to obtain that the gradient operation $\nabla$ is an operation of n simultaneous partial derivatives on the set of functions $F_{n}^{1}$, defined as a mapping from the set of functions $F_{n}^{1}$ to $F_{n}^{n}$:

$$\nabla：F_{n}^{1} \longrightarrow F_{n}^{n}$$

The generalized gradient operation, $\partial$, is defined as the simultaneous operation of n partial derivatives on the set of functions $F_{n}^{m}$.

$$\partial: F_{n}^{m} \longrightarrow F_{n}^{m \times n}$$

The Jacobian matrix is the result obtained by applying the operation $\partial$ to $\textbf{f}$, i.e.

$$\textbf{f} \longmapsto \partial \textbf{f} = (\frac{\partial \textbf{f}}{\partial x_{1}}, \frac{\partial \textbf{f}}{\partial x_{2}}, \dots, \frac{\partial \textbf{f}}{\partial x_{n}})$$

Obtain the Jacobian matrix:

$$J_{f} = \begin{bmatrix}
  \frac{\partial f_{1}}{\partial x_{1}} &\frac{\partial f_{1}}{\partial x_{2}} &\dots &\frac{\partial f_{1}}{\partial x_{n}}  \\
  \frac{\partial f_{2}}{\partial x_{1}} &\frac{\partial f_{2}}{\partial x_{2}} &\dots &\frac{\partial f_{2}}{\partial x_{n}} \\
  \vdots &\vdots &\ddots &\vdots \\
  \frac{\partial f_{m}}{\partial x_{1}} &\frac{\partial f_{m}}{\partial x_{2}} &\dots &\frac{\partial f_{m}}{\partial x_{n}}
\end{bmatrix}$$

Applications of Jacobian matrix: In the forward mode of automatic differentiation, for each forward propagation, we can work out one row of the Jacobian matrix. In the inverse mode of automatic differentiation, for each backward propagation, we can compute a row of the Jacobian matrix.

### Computing the Jacobian Matrix

It is difficult to compute Jacobian matrices efficiently using standard automatic differentiation systems, but MindSpore provides methods that can compute Jacobian efficiently, which are described below.

First we define the function `forecast`, which is a simple linear function with a non-linear activation function.

```python
import time
import mindspore
from mindspore import ops
from mindspore import jacrev, jacfwd, vmap, vjp, jvp, grad
import numpy as np

mindspore.set_seed(1)

def forecast(weight, bias, x):
    return ops.dense(x, weight, bias).tanh()
```

Next, we construct some data: a weight tensor `weight`, a bias tensor `bias`, and an input vector `x`.

```python
D = 16
weight = ops.randn(D, D)
bias = ops.randn(D)
x = ops.randn(D)
```

The function `forecast` does the following mapping transformation on the input vector `x`, $R^{D}\overset{}{\rightarrow}R^{D}$.
MindSpore computes the vector-jacobian product during automatic differentiation. To compute the full Jacobian matrix for the mapping $R^{D}\overset{}{\rightarrow}R^{D}$, we compute it one row at a time using different unit vectors.

```python
def partial_forecast(x):
    return ops.dense(x, weight, bias).tanh()

_, vjp_fn = vjp(partial_forecast, x)

def compute_jac_matrix(unit_vectors):
    jacobian_rows = [vjp_fn(vec)[0] for vec in unit_vectors]
    return ops.stack(jacobian_rows)


unit_vectors = ops.eye(D)
jacobian = compute_jac_matrix(unit_vectors)
print(jacobian.shape)
print(jacobian[0])
```

```text
(16, 16)
[-3.2045446e-05 -1.3530695e-05  1.8671712e-05 -9.6547810e-05
  5.9755850e-05 -5.1343523e-05  1.3528993e-05 -4.6988782e-05
 -4.5517798e-05 -6.1188715e-05 -1.6264191e-04  5.5033437e-05
 -4.3497541e-05  2.2357668e-05 -1.3188722e-04 -3.0677278e-05]
```

In `compute_jac_matrix`, calculating the jacobian matrix using a for loop row by row is not very efficient. MindSpore provides `jacrev` to calculate the jacobian matrix, and the implementation of `jacrev` makes use of `vmap`, which removes the for loop in `compute_jac_matrix` and vectorizes the entire calculation process. The argument `grad_position` to `jacrev` specifies the Jacobian matrix with respect to which the output is computed.

```python
from mindspore import jacrev
jacrev_jacobian = jacrev(forecast, grad_position=2)(weight, bias, x)
assert np.allclose(jacrev_jacobian.asnumpy(), jacobian.asnumpy())
```

The performance of `compute_jac_matrix` and `jacrev` is compared. Typically, `jacrev` has better performance because using `vmap` for vectorization will more fully utilize the hardware to compute multiple sets of data at the same time, reducing the computational overhead for better performance.

Let's write a function that evaluates the performance of both methods on a microsecond scale.

```python
def perf_compution(func, run_times, *args, **kwargs):
    start_time = time.perf_counter()
    for _ in range(run_times):
        func(*args, **kwargs)
    end_time = time.perf_counter()
    cost_time = (end_time - start_time) * 1000000
    return cost_time


run_times = 500
xp = x.copy()
compute_jac_matrix_cost_time = perf_compution(compute_jac_matrix, run_times, xp)
jac_fn = jacrev(forecast, grad_position=2)
jacrev_cost_time = perf_compution(jac_fn, run_times, weight, bias, x)
print(f"compute_jac_matrix run {run_times} times, cost time {compute_jac_matrix_cost_time} microseconds.")
print(f"jacrev run {run_times} times, cost time {jacrev_cost_time} microseconds.")
```

```text
compute_jac_matrix run 500 times, cost time 12942823.04868102 microseconds.
jacrev run 500 times, cost time 909309.7001314163 microseconds.
```

Run `compute_jac_matrix` and `jacrev` 500 times respectively and count the time consumption.

The following calculates the percentage performance improvement of using `jacrev` to compute the Jacobian matrix compared to using `compute_jac_matrix`.

```python
def perf_cmp(first, first_descriptor, second, second_descriptor):
    faster = second
    slower = first
    gain = (slower - faster) / slower
    if gain < 0:
        gain *= -1
    final_gain = gain*100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor}. ")

perf_cmp(compute_jac_matrix_cost_time, "for loop", jacrev_cost_time, "jacrev")
```

```text
 Performance delta: 92.9744 percent improvement with jacrev.
```

It is also possible to compute the Jacobian matrix of the output with respect to the model parameters weight and bias by specifying the parameter `grad_position` of `jacrev`.

```python
jacrev_weight, jacrev_bias = jacrev(forecast, grad_position=(0, 1))(weight, bias, x)
print(jacrev_weight.shape)
print(jacrev_bias.shape)
```

```text
(16, 16, 16)
(16, 16)
```

### Inverse Mode Computation of Jacobian Matrix vs Forward Mode Computation of Jacobian Matrix

MindSpore provides two APIs to compute Jacobian matrices: `jacrev` and `jacfwd`, respectively.

- `jacrev`: Automatic differentiation using inverse mode.
- `jacfwd`: Automatic differentiation using forward mode.

`jacfwd` and `jacrev` are interchangeable, but they perform differently in different scenarios.

In general, if one needs to compute the Jacobian matrix of a function $R^{n}\overset{}{\rightarrow}R^{m}$, `jacfwd` performs better in terms of performance when the size of the output vector of that function is larger than the size of the input vector (i.e., m > n), otherwise `jacrev` performs better in terms of performance.

The following is a non-rigorous argument for this conclusion, in that the forward mode auto-differentiation (computing the Jacobian - vector product) is computed column by column, and in the reverse mode auto-differentiation (computing the vector - Jacobian product) is computed row by row for the Jacobian matrix. Assuming that the size of the Jacobian matrix to be computed is m rows and n columns, we recommend using `jacfwd` to compute the Jacobian matrix column-by-column if m > n, and vice versa if m < n, we recommend using `jacrev` to compute the Jacobian matrix row-by-row.

### Hessian Matrix

Before describing the methods provided by MindSpore to compute the Hessian matrix, the Hessian matrix is first introduced.

The Hessian matrix can be obtained from the composite of the gradient operation $\nabla$ and the generalized gradient operation $\partial$, namely

$$\nabla \circ \partial: F_{n}^{1} \longrightarrow F_{n}^{n} \longrightarrow F_{n \times n}^{n}$$

Applying this composite operation to f yields that

$$f \longmapsto \nabla f \longmapsto J_{\nabla f}$$

The Hessian matrix is obtained:

$$H_{f} = \begin{bmatrix}
  \frac{\partial (\nabla _{1}f)}{\partial x_{1}} &\frac{\partial (\nabla _{1}f)}{\partial x_{2}} &\dots &\frac{\partial (\nabla _{1}f)}{\partial x_{n}}  \\
  \frac{\partial (\nabla _{2}f)}{\partial x_{1}} &\frac{\partial (\nabla _{2}f)}{\partial x_{2}} &\dots &\frac{\partial (\nabla _{2}f)}{\partial x_{n}} \\
  \vdots &\vdots &\ddots &\vdots \\
  \frac{\partial (\nabla _{n}f)}{\partial x_{1}} &\frac{\partial (\nabla _{n}f)}{\partial x_{2}} &\dots &\frac{\partial (\nabla _{n}f)}{\partial x_{n}}
\end{bmatrix} = \begin{bmatrix}
  \frac{\partial ^2 f}{\partial x_{1}^{2}} &\frac{\partial ^2 f}{\partial x_{2} \partial x_{1}} &\dots &\frac{\partial ^2 f}{\partial x_{n} \partial x_{1}}  \\
  \frac{\partial ^2 f}{\partial x_{1} \partial x_{2}} &\frac{\partial ^2 f}{\partial x_{2}^{2}} &\dots &\frac{\partial ^2 f}{\partial x_{n} \partial x_{2}} \\
  \vdots &\vdots &\ddots &\vdots \\
  \frac{\partial ^2 f}{\partial x_{1} \partial x_{n}} &\frac{\partial ^2 f}{\partial x_{2} \partial x_{n}} &\dots &\frac{\partial ^2 f}{\partial x_{n}^{2}}
\end{bmatrix}$$

It is easy to see that the Hessian matrix is a real symmetric matrix.

Application of Hessian matrix: using the Hessian matrix, we can explore the curvature of the neural network at a certain point, providing a numerical basis for whether the training converges.

### Computing the Hessian Matrix

In MindSpore, we can compute the Hessian matrix by any combination of `jacfwd` and `jacrev`.

```python
Din = 32
Dout = 16
weight = ops.randn(Dout, Din)
bias = ops.randn(Dout)
x = ops.randn(Din)

hess1 = jacfwd(jacfwd(forecast, grad_position=2), grad_position=2)(weight, bias, x)
hess2 = jacfwd(jacrev(forecast, grad_position=2), grad_position=2)(weight, bias, x)
hess3 = jacrev(jacfwd(forecast, grad_position=2), grad_position=2)(weight, bias, x)
hess4 = jacrev(jacrev(forecast, grad_position=2), grad_position=2)(weight, bias, x)

np.allclose(hess1.asnumpy(), hess2.asnumpy())
np.allclose(hess2.asnumpy(), hess3.asnumpy())
np.allclose(hess3.asnumpy(), hess4.asnumpy())
```

```text
True
```

### Computing the Batch Jacobian Matrix and the Batch Hessian Matrix

In the examples given above, we are computing the Jacobian matrix of a single output vector with respect to a single input vector. In some cases, you may want to compute the Jacobian matrix of a batch of output vectors with respect to a batch of input vectors, or in other words, given a batch of input vectors with a shape of (b, n) and a function whose mapping relation is $$R^{n}\overset{}{\rightarrow}R^{m}$$. We would expect to get a batch of Jacobian matrices with a shape is (b, m, n).

We can use `vmap` to compute the batch Jacobian matrix.

```python
batch_size = 64
Din = 31
Dout = 33

weight = ops.randn(Dout, Din)
bias = ops.randn(Dout)
x = ops.randn(batch_size, Din)

compute_batch_jacobian = vmap(jacrev(forecast, grad_position=2), in_axes=(None, None, 0))
batch_jacobian = compute_batch_jacobian(weight, bias, x)
print(batch_jacobian.shape)
```

```text
(64, 33, 31)
```

Computing the batch Hessian matrix is similar to computing the batch Jacobian matrix using `vmap`.

```python
hessian = jacrev(jacrev(forecast, grad_position=2), grad_position=2)
compute_batch_hessian = vmap(hessian, in_axes=(None, None, 0))
batch_hessian = compute_batch_hessian(weight, bias, x)
print(batch_hessian.shape)
```

```text
(64, 33, 31, 31)
```

### Computing the Hessian-vector Product

The most straightforward way to compute a Hessian-vector product (hvp) is to compute a complete Hessian matrix and dot-product it with vectors. However, MindSpore provides a better way to compute the Hessian-vector product without computing a complete Hessian matrix. Below we describe two ways to compute the Hessian-Vector product.

- Combine two reverse mode auto-differential.
- Combine reverse mode auto-differentiation with forward mode auto-differentiation.

The following first describes, in MindSpore, how to compute the Hessian-vector product, using a combination of reverse mode auto-differentiation and forward mode auto-differentiation.

```python
def hvp_revfwd(f, inputs, vector):
    return jvp(grad(f), inputs, vector)[1]

def f(x):
    return x.sin().sum()

inputs = ops.randn(128)
vector = ops.randn(128)

result_hvp_revfwd = hvp_revfwd(f, inputs, vector)
print(result_hvp_revfwd.shape)
```

```text
(128,)
```

If forward auto-differentiation does not suffice, we can use a combination of reverse mode auto-differentiation and reverse mode auto-differentiation to compute the Hessian-vector product:

```python
def hvp_revrev(f, inputs, vector):
    _, vjp_fn = vjp(grad(f), *inputs)
    return vjp_fn(*vector)

result_hvp_revrev = hvp_revrev(f, (inputs,), (vector,))
print(result_hvp_revrev[0].shape)
```

```text
(128,)
```

Computing the Hessian-vector product using the two methods above gives the same result.

```python
assert np.allclose(result_hvp_revfwd.asnumpy(), result_hvp_revrev[0].asnumpy())
```
