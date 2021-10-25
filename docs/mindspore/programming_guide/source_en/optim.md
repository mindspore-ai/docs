# Optimization Algorithms

`Ascend` `GPU` `CPU` `Model Development`

<!-- TOC -->

- [Optimization Algorithms](#optimization-algorithms)
    - [Overview](#overview)
    - [Learning Rates](#learning-rates)
        - [dynamic_lr](#dynamic_lr)
        - [learning_rate_schedule](#learning_rate_schedule)
    - [Optimzers](#optimzers)
        - [Usage](#usage)
        - [Built-in Optimizers](#built-in-optimizers)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/optim.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

`mindspore.nn.optim` is a module in the MindSpore framework for implementing various optimization algorithms, including common optimizers and learning rates. In addition, the universal APIs can integrate updated and complex methods into the module.

`mindspore.nn.optim` provides common optimizers for models, such as `SGD`, `ADAM`, and `Momentum`. The optimizer is used to compute and update the gradient. The selection of the model optimization algorithm directly affects the performance of the final model. If the effect is poor, the problem may be caused by the optimization algorithm instead of the feature or model design. In addition, `mindspore.nn` provides the learning rate module. Learning rates are classified into `dynamic_lr` and `learning_rate_schedule`, which are both dynamic learning rates. However, the implementation methods are different. The learning rate is the most important parameter in supervised learning and deep learning. It determines whether the objective function can converge to a local minimum and when it can converge to a minimum. An appropriate learning rate can make the objective function converge to a local minimum in an appropriate time.

> All the following examples support the CPU, GPU, and Ascend environments.

## Learning Rates

### dynamic_lr

The `mindspore.nn.dynamic_lr` module contains the following classes:

- `piecewise_constant_lr` class: computes the learning rate based on the unchanged segment.
- `exponential_decay_lr` class: computes the learning rate based on the exponential decay function.
- `natural_exp_decay_lr` class: computes the learning rate based on the natural exponential decay function.
- `inverse_decay_lr` class: computes the learning rate based on the inverse time attenuation function.
- `cosine_decay_lr` class: computes the learning rate based on the cosine attenuation function.
- `polynomial_decay_lr` class: computes the learning rate based on the polynomial attenuation function.
- `warmup_lr` class: improves the learning rate.

They are different implementations of `dynamic_lr`.

For example, the code example of the `piecewise_constant_lr` class is as follows:

```python
from mindspore.nn import piecewise_constant_lr

def test_dynamic_lr():
    milestone = [2, 5, 10]
    learning_rates = [0.1, 0.05, 0.01]
    lr = piecewise_constant_lr(milestone, learning_rates)
    print(lr)


if __name__ == '__main__':
    test_dynamic_lr()
```

The following information is displayed:

```text
[0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
```

### learning_rate_schedule

The `mindspore.nn.learning_rate_schedule` module has the following classes: `ExponentialDecayLR`, `NaturalExpDecayLR`, `InverseDecayLR`, and `CosineDecayLR`. `PolynomialDecayLR` class and `WarmUpLR` class. They belong to `learning_rate_schedule` but are implemented in different ways. Their meanings are as follows:

- `ExponentialDecayLR` class: computes the learning rate based on the exponential decay function.
- `NaturalExpDecayLR` class: computes the learning rate based on the natural exponential decay function.
- `InverseDecayLR` class: computes the learning rate based on the inverse time attenuation function.
- `CosineDecayLR` class: computes the learning rate based on the cosine attenuation function.
- `PolynomialDecayLR` class: computes the learning rate based on the polynomial attenuation function.
- `WarmUpLR` class: improves the learning rate.

They are different implementations of `learning_rate_schedule`.

For example, the code example of the ExponentialDecayLR class is as follows:

```python
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.nn import ExponentialDecayLR

def test_learning_rate_schedule():
    learning_rate = 0.1    # learning_rate(float) - The initial value of learning rate.
    decay_rate = 0.9    # decay_rate(float) - The decay rate.
    decay_steps = 4    # decay_steps(int) - A value used to calculate decayed learning rate.
    global_step = Tensor(2, mstype.int32)
    exponential_decay_lr = ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
    res = exponential_decay_lr(global_step)
    print(res)


if __name__ == '__main__':
    test_learning_rate_schedule()
```

The following information is displayed:

```text
0.094868325
```

## Optimzers

### Usage

To use `mindspore.nn.optim`, you need to build an `Optimizer` object. This object can maintain the current parameter status and update parameters based on the computed gradient.

- Building

To build an `Optimizer`, you need to give it an iterable that contains the parameters (must be Variable objects) that need to be optimized. Then, you can set the `Optimizer` parameter options, such as the learning rate and weight attenuation.

A code example is as follows:

```python
from mindspore import nn

optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)
optim = nn.Adam(params=net.trainable_params())

optim = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0)

```

- Setting options for each parameter separately

The optimizer also allows you to set options for each parameter separately. Do not pass in the variable directly but pass in the iterable of a dictionary. Each dictionary defines a group of parameters and contains a key, which corresponds to a parameter value. Other keys should be other parameters accepted by the optimizer and will be used to optimize this group of parameters.

You can pass options as keyword parameters, which are used as default values in groups where these options are not overridden. This is useful when you want to change the options of only one parameter group without changing the options of other parameter groups.
Take `SGD` as an example. When you want to determine the learning rate of each layer, run the following command:

```python
from mindspore import nn

optim = nn.SGD([{'params': conv_params, 'weight_decay': 0.01},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}],
               learning_rate=0.1, weight_decay=0.0)

```

This example indicates that when the parameter is conv_params, the weight attenuation is 0.01 and the learning rate is 0.1. When the parameter is no_conv_params, the weight attenuation is 0.0 and the learning rate is 0.01. The learning_rate=0.1 is used for all groups where the learning rate is not set. The same rule applies to weight_deca.

### Built-in Optimizers

Common deep learning optimization algorithms include `SGD`, `Adam`, `Ftrl`, `lazyadam`, `Momentum`, `RMSprop`, `Lars`, `Proximal_ada_grad`, and `lamb`.
In the `mindspore.nn.optim` module, they have corresponding class implementations. For example:

- `SGD`: The default parameter is pure SGD. When the `momentum` parameter is set to a value other than 0, the first-order momentum is considered. After `nesterov` is set to True, the value changes to `NAG`, that is, `Nesterov Accelerated Gradient`. When the gradient is computed, the gradient of the step forward is computed.

- `RMSprop` considers the second-order momentum. Different parameters have different learning rates, that is, adaptive learning rates. `Adagrad` is optimized. Only the second-order momentum within a certain window is considered through exponential smoothing.

- `Adam` considers both first-order momentum and second-order momentum. It can be seen as a further consideration of the first-order momentum based on `RMSprop`.

For example, the code example of `SGD` is as follows:

```python
from mindspore import nn, Model, Tensor
import mindspore.ops as ops
import numpy as np
from mindspore import dtype as mstype
from mindspore import Parameter

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.z = Parameter(Tensor(np.array([1.0], np.float32)))
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

net = Net()
optim = nn.SGD(params=net.trainable_params())

conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

```
