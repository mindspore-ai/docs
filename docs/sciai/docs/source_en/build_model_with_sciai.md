# Building Neural Networks with SciAI

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/sciai/docs/source_en/build_model_with_sciai.md)&nbsp;&nbsp;

SciAI base framework consists of several modules covering network setup, network training, validation and auxiliary functions.

The following examples indicates the fundamental processes in using SciAI to build a neural network model.

> You can download the full sample code here:
> <https://gitee.com/mindspore/mindscience/tree/r0.5/SciAI/tutorial>

## Model Building Basics

The principle of setting up a neural network in ScAI is the same as in [MindSpore](https://www.mindspore.cn/tutorials/en/r2.2/beginner/model.html), but in SciAI it is much easier.

This chapter takes a Multi-Layer Percetron(MLP) as example, introduces how to train a network to solve the following equation.

$$
f(x) = {x_1}^2 + sin(x_2)
$$

For the codes of this part, please refer to the [codes](https://gitee.com/mindspore/mindscience/blob/r0.5/SciAI/tutorial/example_net.py).

### Setup Neural Networks

The following code segment creates a multi-layer perceptron with 2-D input, 1-D output and two 5-D hidden layers.

```python
from sciai.architecture import MLP
from sciai.common.initializer import XavierTruncNormal

net = MLP(layers=[2, 5, 5, 1], weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
```

`MLP` will use normal distribution to initialize the weights, and initialize bias with zeros by default. The activation function is `tanh` by default.

At meantime, `MLP` accepts various initialization method and all [activation functions](https://www.mindspore.cn/docs/en/r2.2/api_python/mindspore.nn.html) provided by MindSpore, as well as those designed for scientific computing.

### Loss Definition

We define the loss function as a sub-class of [Cell](https://www.mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html), and calculate the loss in method `construct`.

```python
from mindspore import nn
from sciai.architecture import MSE

class ExampleLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.mse = MSE()

    def construct(self, x, y_true):
        y_predict = self.network(x)
        return self.mse(y_predict - y_true)

net_loss = ExampleLoss(net)
```

The current loss predicted by `net` can be calculated by calling `net_loss` directly and taking the input `x` as the parameter with the true value `y_true`.

```python
from mindspore import Tensor

x = Tensor([[0.5, 0.5]])
y_true = Tensor([0.72942554])
print("loss value: ", net_loss(x, y_true))
# expected output
...
loss value: 0.3026065
```

### Model Training and Evaluation

Then, by creating instance of trainer class provided by SciAI, we can start training with datasets.
In this case, we randomly sample the equation mentioned above to generate dataset `x_train` and `y_true` for training.

The code segment for training is given as follows, indicating several abilities of SciAI.
The trainer class `TrainCellWithCallBack` is similar to [MindSpore.nn.TrainOneStepCell](https://www.mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.TrainOneStepCell.html),
which needs network `net_loss` and optimizer as parameters and provides callbacks for scientific computation.
Callbacks include printing loss values and time consumption during training and automatic `ckpt` files saving.
The following code wll print the `loss` and time consumption every 100 epochs and save the current model as `ckpt` file every 1000 epochs.
SciAI provides the tool `to_tensor`, which converts multiple numpy data to `Tensor` conveniently.
Use `log_config` to specify the target directory for automatically saving the `TrainCellWithCallBack` callback printouts, as well as those printed by the user using `print_log`.

```python
import numpy as np
from mindspore import nn
from sciai.common import TrainCellWithCallBack
from sciai.context.context import init_project
from sciai.utils import to_tensor, print_log, log_config

# Get the correct platform automatically and set to GRAPH_MODE by default.
init_project()
# Auto log saving
log_config("./logs")

def func(x):
    """The function to be learned to"""
    return x[:, 0:1] ** 2 + np.sin(x[:, 1:2])

optimizer = nn.Adam(net_loss.trainable_params())
trainer = TrainCellWithCallBack(net_loss, optimizer, loss_interval=100, time_interval=100, ckpt_interval=1000)
x_train = np.random.rand(1000, 2)
# Randomly collect ground truth
y_true = func(x_train)
# Convert to Tensor data type
x_train, y_true = to_tensor((x_train, y_true))
for _ in range(10001):
    trainer(x_train, y_true)
print_log("Finished")
```

The expected output is as follows.

```bash
python ./example_net.py
# expected output
...
step: 0, loss: 0.5189553, interval: 2.7039313316345215s, total: 2.7039313316345215s
step: 100, loss: 0.080132075, interval: 0.11984062194824219s, total: 2.8237719535827637s
step: 200, loss: 0.055663396, interval: 0.09104156494140625s, total: 2.91481351852417s
step: 300, loss: 0.032194577, interval: 0.09095025062561035s, total: 3.0057637691497803s
step: 400, loss: 0.015914217, interval: 0.09099435806274414s, total: 3.0967581272125244s
...
Finished
```

When the training of `net` is finished and loss converges, we can use the net to predict the value at `x` by calling `y = net(x)`.
Continue to randomly sample a number of positions `x_val` for validation.

```python
x_val = np.random.rand(5, 2)
y_true = func(x_val)
y_pred = net(to_tensor(x_val)).asnumpy()
print_log("y_true:")
print_log(y_true)
print_log("y_pred:")
print_log(y_pred)
```

The expected output is as follows. After training, the predicted values are close to those obtained by numerical calculation.

```bash
# expected output
y_true:
[[0.34606973]
 [0.70457536]
 [0.90531053]
 [0.84420218]
 [0.48239506]]
y_pred:
[[0.34271246]
 [0.70356864]
 [0.89893466]
 [0.8393946 ]
 [0.47805673]]
```

## Model Building Extension

User can solve more complicated problems with SciAI, for example Physics-Informed Neural Network(PINN). This chapter introduces how to use MLP to solve the following Partial Differential Equation(PDE) with SciAI.

$$
\frac{\partial{f}}{\partial{x}} - 2 \frac{f}{x} + {x}^2 {y}^2 = 0
$$

The boundary conditions are defined to be

$$
f(0) = 0, f(1) = 1.
$$

Under those boundary conditions, the analytic solution of this PDE is

$$
f(x) = \frac{x^2}{0.2 x^5 + 0.8}.
$$

For the codes of this part, please refer to [codes](https://gitee.com/mindspore/mindscience/blob/r0.5/SciAI/tutorial/example_grad_net.py).

### Loss Definition

Similar to the loss definition in the last chapter, the loss should be defined as a child class of [Cell](https://www.mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Cell.html).

The difference is that in this loss function, the partial derivative of the original function needs to be calculated.
SciAI provides tool `operators.grad` for this situation. Through setting the input and output index, we can calculate the derivative of certain inputs w.r.t. certain output.
In this problem, the dimensions of input and output are 1, therefore, we set `input_index` and `output_index` to 0.

```python
from mindspore import nn, ops
from sciai.architecture import MSE, MLP
from sciai.operators import grad

class ExampleLoss(nn.Cell):
    """ Loss definition class"""
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.dy_dx = grad(net=self.network, output_index=0, input_index=0)  # partial differential definition
        self.mse = MSE()

    def construct(self, x, x_bc, y_bc_true):
        y = self.network(x)
        dy_dx = self.dy_dx(x)
        domain_res = dy_dx - 2 * ops.div(y, x) + ops.mul(ops.pow(x, 2), ops.pow(y, 2))  # PDE residual error

        y_bc = self.network(x_bc)
        bc_res = y_bc_true - y_bc  # Boundary conditions residual
        return self.mse(domain_res) + 10 * self.mse(bc_res)
```

### Model Training and Evaluation

Executing training and evaluation by launching the script in the terminal, we can get the following expected outputs. The predictions `y_pred` are close to the true values `y_true`.

```bash
python ./example_grad_net.py
# expected output
...
step: 0, loss: 3.1961572, interval: 3.117840051651001s, total: 3.117840051651001s
step: 100, loss: 1.0862937, interval: 0.23533344268798828s, total: 3.353173494338989s
step: 200, loss: 0.7334847, interval: 0.21307134628295898s, total: 3.566244840621948s
step: 300, loss: 0.5629723, interval: 0.19696831703186035s, total: 3.763213157653809s
step: 400, loss: 0.4133342, interval: 0.20153212547302246s, total: 3.964745283126831s
...
Finished
y_true:
[[0.02245186]
 [0.99459697]
 [0.04027248]
 [0.12594332]
 [0.39779923]]
y_pred:
[[0.02293926]
 [0.99337316]
 [0.03924912]
 [0.12166673]
 [0.4006738 ]]
```