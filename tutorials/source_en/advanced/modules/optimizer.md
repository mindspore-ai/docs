[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/modules/optimizer.md)

# Optimizer

During model training, the optimizer is used to update network parameters. A proper optimizer can effectively reduce the training time and improve model performance.

The most basic optimizer is the stochastic gradient descent (SGD) algorithm. Many optimizers are improved based on the SGD to achieve the target function to converge to the global optimal point more quickly and effectively. The `nn` module in MindSpore provides common optimizers, such as `nn.SGD`, `nn.Adam`, and `nn.Momentum`. The following describes how to configure the optimizer provided by MindSpore and how to customize the optimizer.

![learningrate.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/modules/images/learning_rate.png)

> For details about the optimizer provided by MindSpore, see [Optimizer API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#optimizer).

## nn.optim

### Configuring the Optimizer

When using the optimizer provided by MindSpore, you need to specify the network parameter `params` to be optimized, and then set other main parameters of the optimizer, such as `learning_rate` and `weight_decay`.

If you want to set options for different network parameters separately, for example, set different learning rates for convolutional and non-convolutional parameters, you can use the parameter grouping method to set the optimizer.

#### Parameter Configuration

When building an optimizer instance, you need to use the optimizer parameter `params` to configure the weights to be trained and updated on the model network.

`Parameter` contains a Boolean class attribute `requires_grad`, which is used to indicate whether network parameters in the model need to be updated. The default value of `requires_grad` of most network parameters is True, while the default value of `requires_grad` of a few network parameters is False, for example, `moving_mean` and `moving_variance` in BatchNorm.

The `trainable_params` method in MindSpore shields the attribute whose `requires_grad` is False in `Parameter`. When configuring the input parameter `params` for the optimizer, you can use the `net.trainable_params()` method to specify the network parameters to be optimized and updated.

```python
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.param = Parameter(Tensor(np.array([1.0], np.float32)), 'param')

    def construct(self, x):
        x = self.conv(x)
        x = x * self.param
        out = ops.matmul(x, x)
        return out

net = Net()

# Parameters to be updated for the configuration optimizer
optim = nn.Adam(params=net.trainable_params())
print(net.trainable_params())
```

```text
[Parameter (name=param, shape=(1,), dtype=Float32, requires_grad=True), Parameter (name=conv.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)]
```

You can manually change the default value of the `requires_grad` attribute of `Parameter` in the network weight to determine which parameters need to be updated.

As shown in the following example, use the `net.get_parameters()` method to obtain all parameters on the network and manually change the `requires_grad` attribute of the convolutional parameter to False. During the training, only non-convolutional parameters are updated.

```python
conv_params = [param for param in net.get_parameters() if 'conv' in param.name]
for conv_param in conv_params:
    conv_param.requires_grad = False
print(net.trainable_params())
optim = nn.Adam(params=net.trainable_params())
```

```text
[Parameter (name=param, shape=(1,), dtype=Float32, requires_grad=True)]
```

#### Learning Rate

As a common hyperparameter in machine learning and deep learning, the learning rate has an important impact on whether the target function can converge to the local minimum value and when to converge to the minimum value. If the learning rate is too high, the target function may fluctuate greatly and it is difficult to converge to the optimal value. If the learning rate is too low, the convergence process takes a long time. In addition to setting a fixed learning rate, MindSpore also supports setting a dynamic learning rate. These methods can significantly improve the convergence efficiency on a deep learning network.

**Fixed Learning Rate**:

When a fixed learning rate is used, the `learning_rate` input by the optimizer is a floating-point tensor or scalar tensor.

Take `nn.Momentum` as an example. The fixed learning rate is 0.01. The following is an example:

```python
# Set the learning rate to 0.01.
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)
```

**Dynamic Learning Rate**:

`mindspore.nn` provides the dynamic learning rate module, which is classified into the Dynamic LR function and LearningRateSchedule class. The Dynamic LR function pre-generates a learning rate list whose length is `total_step` and transfers the list to the optimizer for use. During training, the value of the ith learning rate is used as the learning rate of the current step in step `i`. The value of `total_step` cannot be less than the total number of training steps. The LearningRateSchedule class transfers the instance to the optimizer, and the optimizer computes the current learning rate based on the current step.

- Dynamic LR function

    Currently, the [Dynamic LR function](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-lr-function) can compute the learning rate (`nn.cosine_decay_lr`) based on the cosine decay function, the learning rate (`nn.exponential_decay_lr`) based on the exponential decay function, the learning rate (`nn.inverse_decay_lr`) based on the counterclockwise decay function, and the learning rate (`nn.natural_exp_decay_lr`) based on the natural exponential decay function, the piecewise constant learning rate (`nn.piecewise_constant_lr`), the learning rate (`nn.polynomial_decay_lr`) based on the polynomial decay function, and the warm-up learning rate (`nn.warmup_lr`).

    The following uses `nn.piecewise_constant_lr` as an example:

    ```python
    milestone = [1, 3, 10]
    learning_rates = [0.1, 0.05, 0.01]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)

    # Print the learning rate.
    print(lr)

    net = Net()
    # The optimizer sets the network parameters to be optimized and the piecewise constant learning rate.
    optim = nn.SGD(net.trainable_params(), learning_rate=lr)
    ```

    ```text
    [0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    ```

- LearningRateSchedule Class

    Currently, the [LearningRateSchedule class](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#learningrateschedule-class) can compute the learning rate (`nn.CosineDecayLR`) based on the cosine decay function, the learning rate (`nn.ExponentialDecayLR`) based on the exponential decay function, the learning rate (`nn.InverseDecayLR`) based on the counterclockwise decay function, the learning rate (`nn.NaturalExpDecayLR`) based on the natural exponential decay function, the learning rate (`nn.PolynomialDecayLR`) based on the polynomial decay function, and warm-up learning rate (`nn.WarmUpLR`).

    In the following example, the learning rate `nn.ExponentialDecayLR` is computed based on the exponential decay function.

    ```python
    learning_rate = 0.1  # Initial value of the learning rate
    decay_rate = 0.9     # Decay rate
    decay_steps = 4      #Number of decay steps
    step_per_epoch = 2

    exponential_decay_lr = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)

    for i in range(decay_steps):
        step = Tensor(i, mindspore.int32)
        result = exponential_decay_lr(step)
        print(f"step{i+1}, lr:{result}")

    net = Net()

    # The optimizer sets the learning rate and computes the learning rate based on the exponential decay function.
    optim = nn.Momentum(net.trainable_params(), learning_rate=exponential_decay_lr, momentum=0.9)
    ```

    ```text
    step1, lr:0.1
    step2, lr:0.097400375
    step3, lr:0.094868325
    step4, lr:0.09240211
    ```

#### Weight Decay

Weight decay, also referred to as L2 regularization, is a method for mitigating overfitting of a deep neural network.

Generally, the value range of `weight_decay` is $ [0,1) $, and the default value is 0.0, indicating that the weight decay policy is not used.

```python
net = Net()
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.01,
                        momentum=0.9, weight_decay=0.9)
```

In addition, MindSpore supports dynamic weight decay. In this case, `weight_decay` is a customized Cell. During training, the optimizer calls the instance of the Cell and transfers `global_step` to compute the `weight_decay` value of the current step. `global_step` is an internally maintained variable. The value of `global_step` increases by 1 each time a step is trained. The following is an example of exponential decay during training.

```python
from mindspore.nn import Cell
from mindspore import ops, nn
import mindspore as ms

class ExponentialWeightDecay(Cell):

    def __init__(self, weight_decay, decay_rate, decay_steps):
        super(ExponentialWeightDecay, self).__init__()
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def construct(self, global_step):
        # The `construct` can have only one input. During training, the global step is automatically transferred for computation.
        p = global_step / self.decay_steps
        return self.weight_decay * ops.pow(self.decay_rate, p)

net = Net()

weight_decay = ExponentialWeightDecay(weight_decay=0.0001, decay_rate=0.1, decay_steps=10000)
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.01,
                        momentum=0.9, weight_decay=weight_decay)
```

#### Hyperparameter Grouping

The optimizer can also set options for different parameters separately. In this case, a dictionary list is transferred instead of variables. Each dictionary corresponds to a group of parameter values. Available keys in the dictionary include `params`, `lr`, `weight_decay`, and `grad_centralizaiton`, and `value` indicates the corresponding value.

`params` is mandatory, and other parameters are optional. If `params` is not configured, the parameter values set when the optimizer is defined are used. During grouping, the learning rate can be a fixed learning rate or a dynamic learning rate, and `weight_decay` can be a fixed value.

In the following example, different learning rates and weight decay parameters are set for convolutional and non-convolutional parameters.

```python
net = Net()

# Convolutional parameter
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
# Non-convolutional parameter
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))

# Fixed learning rate
fix_lr = 0.01

# Computation of Learning Rate Based on Polynomial Decay Function
polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1,      # Initial learning rate
                                           end_learning_rate=0.01, # Final the learning rate
                                           decay_steps=4,          #Number of decay steps
                                           power=0.5)              # Polynomial power

# The convolutional parameter uses a fixed learning rate of 0.001, and the weight decay is 0.01.
# The non-convolutional parameter uses a dynamic learning rate, and the weight decay is 0.0.
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                {'params': no_conv_params, 'lr': polynomial_decay_lr}]

optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
```

> Except a few optimizers (such as AdaFactor and FTRL), MindSpore supports grouping of learning rates. For details, see [Optimizer API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#optimizer).

### Customized Optimizer

In addition to the optimizers provided by MindSpore, you can customize optimizers.

When customizing an optimizer, you need to inherit the optimizer base class [nn.Optimizer](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Optimizer.html#mindspore.nn.Optimizer) and rewrite the `__init__` and `construct` methods to set the parameter update policy.

The following example implements the customized optimizer Momentum (SGD algorithm with momentum):

$$ v_{t+1} = v_t×u+grad \tag{1} $$

$$p_{t+1} = p_t - lr*v_{t+1} \tag{2} $$

$grad$, $lr$, $p$, $v$, and $u$ respectively represent a gradient, a learning rate, a weight parameter, a momentum parameter, and an initial speed.

```python
class Momentum(nn.Optimizer):
    """Define the optimizer."""
    def __init__(self, params, learning_rate, momentum=0.9):
        super(Momentum, self).__init__(learning_rate, params)
        self.momentum = Parameter(Tensor(momentum, ms.float32), name="momentum")
        self.moments = self.parameters.clone(prefix="moments", init="zeros")

    def construct(self, gradients):
        """The input of construct is gradient. Gradients are automatically transferred during training."""
        lr = self.get_lr()
        params = self.parameters # Weight parameter to be updated

        for i in range(len(params)):
            # Update the moments value.
            ops.assign(self.moments[i], self.moments[i] * self.momentum + gradients[i])
            update = params[i] - self.moments[i] * lr # SGD algorithm with momentum
            ops.assign(params[i], update)
        return params

net = Net()
# Set the parameter to be optimized and the learning rate of the optimizer to 0.01.
opt = Momentum(net.trainable_params(), 0.01)
```

`mindSpore.ops` also encapsulates optimizer operators for users to define optimizers, such as `ops.ApplyCenteredRMSProp`, `ops.ApplyMomentum`, and `ops.ApplyRMSProp`. The following example uses the `ApplyMomentum` operator to customize the optimizer Momentum:

```python
class Momentum(nn.Optimizer):
    """Define the optimizer."""
    def __init__(self, params, learning_rate, momentum=0.9):
        super(Momentum, self).__init__(learning_rate, params)
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.momentum = momentum
        self.opt = ops.ApplyMomentum()

    def construct(self, gradients):
        # Weight parameter to be updated
        params = self.parameters
        success = None
        for param, mom, grad in zip(params, self.moments, gradients):
            success = self.opt(param, mom, self.learning_rate, grad, self.momentum)
        return success

net = Net()
# Set the parameter to be optimized and the learning rate of the optimizer to 0.01.
opt = Momentum(net.trainable_params(), 0.01)
```

## nn.optim_ex

In addition to the optimizer within the `mindspore.nn.optim` module mentioned above, MindSpore also provides an experimental optimizer module, `mindspore.experimental.optim`, which is designed to extend the function of the optimizer.

> The `mindspore.experimental.optim` module is still under development. Currently the optimizer for this module is only available for functional programming scenarios and only adapts to the dynamic learning rate class under [mindspore.experimental.lr_scheduler](https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class).

Usage differences:

| Parameters   | nn.optim | experimental.optim | Functions    |
|-------|----------| ------------|-------|
| Parameter configuration (hyperparameter not grouped) | Configure input to be `params` | Configure input to be `params`  | The configuration and function is the same in normal scenarios, and passing in `net.trainable_params` is sufficient.|
| Learning rate   | Configure input to be `learning_rate` | Configure input to be `lr` |  For configuration and function difference for dynamic learning rate scenarios, see [Dynamic Learning Rate](#learning-rate-1) for details|
| Weight decay  | Configure input to be `weight_decay` | Configure input to be `weight_decay` | For different dynamic weight_decay scenarios configuration, see [weight_decay](#weight-decay-1) for details.|
| Hyperparameter grouping  | Configure input to be `params`, passing in the set of parameters dict | Configure input to be `params`, passing in the set of parameters dict | In the grouping scenario, i.e., when `params` is a dict, the functions are different, see [hyperparameter grouping](#hyperparameter-grouping-1) for details.|

In addition to the above similarities and differences, `mindspore.experimental.optim` also supports [Viewing Parameter Groups](#viewing-optimizer-configuration), [Modifying Optimizer Parameters during Running](#modifying-optimizer-parameters-during-running), and other features, as detailed below.

### Configuring Optimizer

#### Parameter Configuration

In normal scenarios, the parameters are configured in the same way as for `mindspore.nn.optim`, passing in `net.trainable_params`.

#### Learning Rate

**Fixed Learning Rate**:

Configured in the same way as the fixed learning rates of `mindspore.nn.optim`.

**Dynamic Learning Rate**:

The dynamic learning rate module is provided under `mindspore.experimental.lr_scheduler` for use with `mindspore.experimental.optim` and the usage way is different from that of `mindspore.nn.optim`:

`mindspore.nn.optim`: Pass a list or instance of dynamic learning rates to the optimizer input `learning_rate`, as used in [DynamicLR function](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-lr-function) and [LearningRateSchedule class](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#learningrateschedule-class).

`mindspore.experimental.optim`: Pass the optimizer instance to the input `optimizer` of the dynamic learning rate class, as used in [LRScheduler class](https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class).

The `LRScheduler` also provides two ways of obtaining the learning rate:

`get_lr`. Taking `StepLR` as an example, the learning rate can be obtained manually using `scheduler.get_lr()` directly during the training process.

```python
net = Net()
optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.1)
scheduler = nn.StepLR(optimizer, step_size=30, gamma=0.1)
print(scheduler.get_last_lr())
```

```text
[Tensor(shape=[], dtype=Float32, value= 0.1)]
```

Set the input `verbose` to True. Taking `StepLR` as an example, during the training process, when the training period reaches `step_size`, the learning rate will decay `gamma`. Assuming that the value of the learning rate after the decay becomes 0.01, "Adjusting learning rate of group 0 to 0.1." will automatically be displayed on the screen.

```python
net = Net()
optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.1)
scheduler = nn.StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
```

```text
Adjusting learning rate of group 0 to 0.1.
```

#### Weight Decay

`mindspore.nn.optim`: `weight_decay` supports int and float types, and also supports Cell type for dynamic weight_decay scenarios.

`mindspore.nn.optim_ex`: `weight_decay` data type only supports for int and float types, but the user is supported to manually modify the value of weight_decay in PyNative mode.

#### Hyperparameter Grouping

`mindspore.nn.optim`: Specific key groupings are supported: "params", "lr", "weight_decay" and "grad_centralizaiton", see [above](#hyperparameter-grouping) for details on how to use them.

`mindspore.nn.optim_ex`: Supports all optimizer parameter groupings.

Code Example:

```python
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(
    filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [
    {'params': conv_params, 'weight_decay': 0.01, 'lr': 0.9, "amsgrad": True},
    {'params': no_conv_params, 'lr': 0.66, "eps": 1e-6, "betas": (0.8, 0.88)}]
optimizer = nn.optim_ex.Adam(params=group_params, lr=0.01)
```

#### Viewing Optimizer Configuration

**Use the `param_group` attribute to view parameter groups**:

Code Example:

```python
print(optimizer.param_groups)
```

```text
[{'params': [Parameter (name=conv.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)], 'weight_decay': 0.01, 'lr': Parameter (name=learning_rate_group_0, shape=(), dtype=Float32, requires_grad=True), 'amsgrad': True, 'betas': (0.9, 0.999), 'eps': 1e-08, 'maximize': False}, {'params': [Parameter (name=param, shape=(1,), dtype=Float32, requires_grad=True)], 'lr': Parameter (name=learning_rate_group_1, shape=(), dtype=Float32, requires_grad=True), 'eps': 1e-06, 'betas': (0.8, 0.88), 'weight_decay': 0.0, 'amsgrad': False, 'maximize': False}]
```

As you can see from the above output, the learning rate in the optimizer parameter group is `Parameter`. `Parameter` in mindspore does not display the parameter value natively, and you can view the parameter value by using `.value()`. It can use `get_lr` of `mindspore.nn.LRScheduler` from Dynamic Learning Rate Module `mindspore.nn.LRScheduler` as described above or set `verbose=True`.

```python
print(optimizer.param_groups[1]["lr"].value())
```

```text
0.66
```

**Printing Optimizer Instances Directly to View Parameter Groups**:

```python
print(optimizer)
```

```text
Adam (
Parameter Group 0
    amsgrad: True
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.9
    maximize: False
    weight_decay: 0.01

Parameter Group 1
    amsgrad: False
    betas: (0.8, 0.88)
    eps: 1e-06
    lr: 0.66
    maximize: False
    weight_decay: 0.0
)
```

### Modifying Optimizer Parameters during Running

#### Modifying Learning Rate during Running

The learning rate in `mindspore.nn.optim_ex` is `Parameter`, in addition to the dynamic modification of the learning rate through the dynamic learning rate module `mindspore.nn.lr_scheduler` as described above, the modification of the learning rate using the `assign` assignment is also supported.

For example, in the sample below, in the training step, set the learning rate of 1st parameter group in the optimizer to be adjusted to 0.01 if the change in the loss value compared to the previous step is less than 0.1:

```python
net = Net()
loss_fn = nn.MAELoss()
optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.1)
scheduler = nn.StepLR(optimizer, step_size=10, gamma=0.5)
last_step_loss = 0.1

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    if ops.abs(loss - last_step_loss) < 0.1:
        ops.assign(optimizer.param_groups[1]["lr"], Tensor(0.01))
    return loss
```

#### Modifying Optimizer Parameters other than lr during Running

> Currently, only PyNative mode supports modifying other optimizer parameters during running, and modifications in Graph mode will not take effect or report errors.

In the following sample, in the training step, set the `weight_decay` of 1st parameter group in the optimizer to be adjusted to 0.02 if the change in the loss value compared to the previous step is less than 0.1:

```python
net = Net()
loss_fn = nn.MAELoss()
optimizer = nn.optim_ex.Adam(net.trainable_params(), lr=0.1)
scheduler = nn.StepLR(optimizer, step_size=10, gamma=0.5)
last_step_loss = 0.1

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    if ops.abs(loss - last_step_loss) < 0.1:
        optimizer.param_groups[1]["weight_decay"] = 0.02
    return loss
```

### Customized Optimizer

In the same way as the [Customized Optimizer](#customized-optimizer) above, a custom optimizer can also inherit from the optimizer base class [experimental.optim.Optimizer](https://www.mindspore.cn/docs/en/master/api_python/experimental/optim/mindspore.experimental.optim.Optimizer.html) and override the `__init__` method and `construct` method to set your own parameter update strategy.

