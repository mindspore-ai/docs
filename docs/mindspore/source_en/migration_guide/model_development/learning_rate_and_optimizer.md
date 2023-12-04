# Learning Rate and Optimizer

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/model_development/learning_rate_and_optimizer.md)

Before reading this chapter, please read the official MindSpore tutorial [Optimizer](https://mindspore.cn/tutorials/en/master/advanced/modules/optimizer.html).

Here is an introduction to some special ways of using MindSpore optimizer and the principle of learning rate decay strategy.

## Optimizer Comparison

### Optimizer Support Differences

A comparison of the similarities and differences between the optimizers supported by both PyTorch and MindSpore is detailed in the [API mapping table](https://mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#torch-optim). Optimizers not supported in MindSpore at the moment: LBFGS, NAdam, RAdam.

### Optimizer Execution and Usage Differences

When PyTorch executes the optimizer in a single step, it is usually necessary to manually execute the `zero_grad()` method to set the historical gradient to 0 (or None), then use `loss.backward()` to calculate the gradient of the current training step, and finally call the `step()` method of the optimizer to update the network weights;

The use of the optimizer in MindSpore requires only a direct calculation of the gradients and then uses `optimizer(grads)` to perform the update of the network weights.

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
import mindspore
from mindspore import nn

optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

</pre>
</td>
</tr>
</table>
</div>

### Hyperparameter Differences

#### Hyperparameter Names

Similarities and differences between network weight and learning rate parameter names:

| Parameters   | PyTorch | MindSpore | Differences    |
|------|---------| --------- |-------|
| network weight | params  | params      | The parameters are the same |
| learning rate  | lr      | learning_rate      | The parameters are different |

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
from mindspore import nn

optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)
```

</pre>
</td>
</tr>
</table>
</div>

#### Hyperparameter Configuration Methods

- The parameters are not grouped:

    The data types of the `params` different: input types in PyTorch are `iterable(Tensor)` and `iterable(dict)`, which support iterator types, while input types in MindSpore are `list(Parameter)`, `list(dict)`, which do not support iterators.

    Other hyperparameter configurations and support differences are detailed in the [API mapping table](https://mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#torch-optim).

- The parameters are grouped:

    PyTorch supports all parameter groupings. MindSpore supports certain key groupings: "params", "lr", "weight_decay", "grad_centralization", "order_params".

    <div class="wy-table-responsive">
    <table class="colwidths-auto docutils align-default">
    <tr>
    <td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
    </tr>
    <tr>
    <td style="vertical-align:top"><pre>

    ```python
    optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
    ```

    </pre>
    </td>
    <td style="vertical-align:top"><pre>

    ```python
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': 0.02},
            {'params': no_conv_params}]

    optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9)
    ```

    </pre>
    </td>
    </tr>
    </table>
    </div>

#### Runtime Hyperparameter Modification

PyTorch supports modifying arbitrary optimizer parameters during training, and provides `LRScheduler` for dynamically modifying the learning rate;

MindSpore currently does not support modifying optimizer parameters during training, but provides a way to modify the learning rate and weight decay. See the [Learning Rate Strategy Comparison](#learning-rate-strategy-comparison) and [Weight Decay](#weight-decay) sections for details.

### Weight Decay

Modify `weight_decay` in PyTorch is as below.

Implement dynamic weight decay in MindSpore: Users can inherit the class of `Cell` custom dynamic weight decay and pass it into the optimizer.

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch.nn import optim

optimizer = optim.SGD(param_groups, lr=0.01, weight_decay=0.1)
decay_factor = 0.1
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] *= decay_factor
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
class ExponentialWeightDecay(Cell):

    def __init__(self, weight_decay, decay_rate, decay_steps):
        super(ExponentialWeightDecay, self).__init__()
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def construct(self, global_step):
        p = global_step / self.decay_steps
        return self.weight_decay * ops.pow(self.decay_rate, p)

weight_decay = ExponentialWeightDecay(weight_decay=0.1, decay_rate=0.1, decay_steps=10000)
optimizer = nn.SGD(net.trainable_params(), weight_decay=weight_decay)
```

</pre>
</td>
</tr>
</table>
</div>

### Saving and Loading Optimizer State

PyTorch optimizer module provides `state_dict()` for viewing and saving the optimizer state, and `load_state_dict` for loading the optimizer state.

MindSpore optimizer module is inherited from `Cell`. The optimizer is saved and loaded in the same way as the network is saved and loaded, usually in conjunction with `save_checkpoint` and `load_checkpoint`.

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# Optimizer saving:
# Use torch.save() to save the obtained `state_dict` to a pkl file
optimizer = optim.SGD(param_groups, lr=0.01)
torch.save(optimizer.state_dict(), save_path)
```

```python
# Optimizer loading:
# Use torch.load() to load the saved `state_dict`, and then
# use `load_state_dict` to load the obtained `state_dict` into the optimizer
optimizer = optim.SGD(param_groups, lr=0.01)
state_dict = torch.load(save_path)
optimizer.load_state_dict(state_dict)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# Optimizer saving:
# Use mindspore.save_checkpoint() to save the optimizer instance to a ckpt file
optimizer = nn.SGD(param_groups, lr=0.01)
state_dict = mindspore.save_checkpoint(opt, save_path)
```

```python
# Optimizer loading:
# Use mindspore.load_checkpoint() to load the saved ckpt file, and then
# use `load_param_into_net` to load the obtained `param_dict` into the optimizer
optimizer = nn.SGD(param_groups, lr=0.01)
param_dict = mindspore.load_checkpoint(save_path)
mindspore.load_param_into_net(opt, param_dict)
```

</pre>
</td>
</tr>
</table>
</div>

## Learning Rate Strategy Comparison

### Dynamic Learning Rate Differences

The `LRScheduler` class is defined in PyTorch to manage the learning rate. To use dynamic learning rates, pass an `optimizer` instance into the `LRScheduler` subclass, call `scheduler.step()` in a loop to perform learning rate modifications, and synchronize the changes to the optimizer.

There are two implementations of dynamic learning rates in MindSpore, `Cell` and `list`. Both types of dynamic learning rates are used in the same way and are passed into the optimizer after instantiation is complete. The former computes the learning rate at each step in the internal `construct`, while the latter pre-generates the learning rate list directly according to the computational logic, and updates the learning rate internally during the training process. Please refer to [Dynamic Learning Rate](https://mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-learning-rate) for details.

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01, decay_steps=4, power=0.5)
optim = nn.Momentum(params, learning_rate=polynomial_decay_lr, momentum=0.9, weight_decay=0.0)

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

</pre>
</td>
</tr>
</table>
</div>

### Custom Learning Rate Differences

PyTorch dynamic learning rate module, `LRScheduler`, provides a `LambdaLR` interface for custom learning rate adjustment rules, which can be specified by passing lambda expressions or custom functions.

MindSpore does not provide a similar lambda interface. Custom learning rate adjustment rules can be implemented through custom functions or custom `LearningRateSchedule`.

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
lbd = lambda epoch: epoch // 5
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lbd)

for epoch in range(20):
    train(...)
    validate(...)
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# Method 1: Define the calculation logic specified by the python function,
# and return a list of learning rates
def dynamic_lr(lr, total_step, step_per_epoch):
    lrs = []
    for i in range(total_step):
        current_epoch = i // step_per_epoch
        factor = current_epoch // 5
        lrs.append(lr * factor)
    return lrs

decay_lr = dynamic_lr(lr=0.01, total_step=200, step_per_epoch=10)
optim = nn.SGD(params, learning_rate=decay_lr)
```

```python
# Method 2: Inherit `LearningRateSchedule` and
# define the change policy in the `construct` method
class DynamicDecayLR(LearningRateSchedule):
    def __init__(self, lr, step_per_epoch):
        super(DynamicDecayLR, self).__init__()
        self.lr = lr
        self.step_per_epoch = step_per_epoch
        self.cast = P.Cast()

    def construct(self, global_step):
        current_epoch = self.cast(global_step, mstype.float32) // step_per_epoch
        return self.learning_rate * (current_epoch // 5)

decay_lr = DynamicDecayLR(lr=0.01, step_per_epoch=10)
optim = nn.SGD(params, learning_rate=decay_lr)
```

</pre>
</td>
</tr>
</table>
</div>

### Obatining the Learning Rate

PyTorch:

- In the fixed learning rate scenario, the learning rate is usually viewed and printed by `optimizer.state_dict()`. For example, when parameters are grouped, use `optimizer.state_dict()['param_groups'][n]['lr']` for the nth parameter group, and use `optimizer.state_dict()['param_groups'][0]['lr']` when the parameters are not grouped;

- In the dynamic learning rate scenario, you can use the `get_lr` method of the `LRScheduler` to get the current learning rate or the `print_lr` method to print the learning rate.

MindSpore:

- The interface to view the learning rate directly is not provided at present, and the problem will be fixed in the subsequent version.

### Learning Rate Update

PyTorch:

PyTorch provides the `torch.optim.lr_scheduler` package for dynamically modifying LR. When using the package, you need to explicitly call `optimizer.step()` and `scheduler.step()` to update LR. For details, see [How Do I Adjust the Learning Rate](https://pytorch.org/docs/1.12/optim.html#how-to-adjust-learning-rate).

MindSpore:

The learning rate of MindSpore is packaged in the optimizer. Each time the optimizer is invoked, the learning rate update step is automatically updated..

## Parameters Grouping

MindSpore optimizer supports some special operations, such as different learning rates (lr), weight_decay and gradient_centralization strategies can be set for all trainable parameters in the network. For example:

```python
from mindspore import nn

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            nn.Conv2d(3, 12, kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(12, 4, kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.pool = nn.AdaptiveMaxPool2d((5, 5))
        self.fc = nn.Dense(100, 10)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view((-1, 100))
        out = nn.Dense(x)
        return out

def params_not_in(param, param_list):
    # Use the Parameter id to determine if param is not in the param_list
    param_id = id(param)
    for p in param_list:
        if id(p) == param_id:
            return False
    return True

net = Network()
trainable_param = net.trainable_params()
conv_weight, bn_weight, dense_weight = [], [], []
for _, cell in net.cells_and_names():
    # Determine what the API is and add the corresponding parameters to the different lists
    if isinstance(cell, nn.Conv2d):
        conv_weight.append(cell.weight)
    elif isinstance(cell, nn.BatchNorm2d):
        bn_weight.append(cell.gamma)
        bn_weight.append(cell.beta)
    elif isinstance(cell, nn.Dense):
        dense_weight.append(cell.weight)

other_param = []
# The parameters in all groups cannot be duplicated, and the intersection between groups is all the parameters that need to be updated
for param in trainable_param:
    if params_not_in(param, conv_weight) and params_not_in(param, bn_weight) and params_not_in(param, dense_weight):
        other_param.append(param)

group_param = [{'order_params': trainable_param}]
# The parameter list for each group cannot be empty

if conv_weight:
    conv_weight_lr = nn.cosine_decay_lr(0., 1e-3, total_step=1000, step_per_epoch=100, decay_epoch=10)
    group_param.append({'params': conv_weight, 'weight_decay': 1e-4, 'lr': conv_weight_lr})
if bn_weight:
    group_param.append({'params': bn_weight, 'weight_decay': 0., 'lr': 1e-4})
if dense_weight:
    group_param.append({'params': dense_weight, 'weight_decay': 1e-5, 'lr': 1e-3})
if other_param:
    group_param.append({'params': other_param})

opt = nn.Momentum(group_param, learning_rate=1e-3, weight_decay=0.0, momentum=0.9)
```

The following points need to be noted:

1. The list of parameters for each group cannot be empty.
2. Use the values set in the optimizer if `weight_decay` and `lr` are not set, and use the values in the grouping parameter dictionary if they are set.
3. `lr` in each group can be static or dynamic, but cannot be regrouped.
4. `weight_decay` in each group needs to be a conforming floating point number.
5. The parameters in all groups cannot be duplicated, and the intersection between groups is all the parameters that need to be updated.

## MindSpore Learning Rate Decay Strategy

During the training process, MindSpore learning rate is in the form of parameters in the network. Before executing the optimizer to update the trainable parameters in network, MindSpore will call [get_lr](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Optimizer.html#mindspore.nn.Optimizer.get_lr) to get the value of the learning rate needed for the current step.

MindSpore learning rate supports static, dynamic, and grouping, where the static learning rate is a Tensor in float32 type in the network.

There are two types of dynamic learning rates, one is a Tensor in the network, with the length of the total number of steps of training and in float32 type, such as [Dynamic LR function](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#dynamic-lr-function). There is `global_step` in the optimizer, and the parameter will be +1 for every optimizer update. MindSpore will internally get the learning rate value of the current step based on the parameters `global_step` and `learning_rate`.

The other one is the one that generates the value of learning rate by composition, such as [LearningRateSchedule class](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#learningrateschedule-class).

The grouping learning rate is as described in parameter grouping in the previous section.

Because the learning rate of MindSpore is a parameter, we can also modify the value of learning rate during training by assigning values to `learning_rate` parameter, as in [LearningRateScheduler Callback](https://www.mindspore.cn/docs/zh-CN/master/_modules/mindspore/train/callback/_lr_scheduler_callback.html#LearningRateScheduler). This method only supports static learning rates passed into the optimizer. The key code is as follows:

```python
import mindspore as ms
from mindspore import ops, nn

net = nn.Dense(1, 2)
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
print(optimizer.learning_rate.data.asnumpy())
new_lr = 0.01
# Rewrite the value of the learning_rate parameter
ops.assign(optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))
print(optimizer.learning_rate.data.asnumpy())
```

Outputs:

```text
0.1
0.01
```
