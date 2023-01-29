# Learning Rate and Optimizer

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/migration_guide/model_development/learning_rate_and_optimizer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

Before reading this chapter, please read the official MindSpore tutorial [Optimizer](https://mindspore.cn/tutorials/en/r2.0.0-alpha/advanced/modules/optimizer.html).

The chapter of official tutorial optimizer in MindSpore is already detailed, so here is an introduction to some special ways of using MindSpore optimizer and the principle of learning rate decay strategy.

## Parameters Grouping

MindSpore optimizer supports some special operations, such as different learning rates (lr), weight_decay and gradient_centralization strategies can be set for all trainable parameters in the network. For example:

```python
from mindspore import nn

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            nn.Conv2d(3, 12, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(12, 4, kernel_size=3, pad_mode="pad", padding=1),
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

During the training process, MindSpore learning rate is in the form of parameters in the network. Before executing the optimizer to update the trainable parameters in network, MindSpore will call [get_lr](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.Optimizer.html#mindspore.nn.Optimizer.get_lr) to get the value of the learning rate needed for the current step.

MindSpore learning rate supports static, dynamic, and grouping, where the static learning rate is a Tensor in float32 type in the network.

There are two types of dynamic learning rates, one is a Tensor in the network, with the length of the total number of steps of training and in float32 type, such as [Dynamic LR function](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.nn.html#dynamic-lr-function). There is `global_step` in the optimizer, and the parameter will be +1 for every optimizer update. MindSpore will internally get the learning rate value of the current step based on the parameters `global_step` and `learning_rate`.

The other one is the one that generates the value of learning rate by composition, such as [LearningRateSchedule class](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.nn.html#learningrateschedule-class).

The grouping learning rate is as described in parameter grouping in the previous section.

Because the learning rate of MindSpore is a parameter, we can also modify the value of learning rate during training by assigning values to `learning_rate` parameter, as in [LearningRateScheduler Callback](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/_modules/mindspore/train/callback/_lr_scheduler_callback.html#LearningRateScheduler). This method only supports static learning rates passed into the optimizer. The key code is as follows:

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

```text
0.1
0.01
```
