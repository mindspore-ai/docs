# 学习率与优化器

在阅读本章节之前，请先阅读MindSpore官网教程[优化器](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/network/optim.html)。

MindSpore官网教程优化器章节的内容已经很详细了，这里就MindSpore的优化器的一些特殊使用方式和学习率衰减策略的原理做一个介绍。

## 参数分组

MindSpore的优化器支持一些特别的操作，比如对网络里所有的可训练的参数可以设置不同的学习率（lr ）、权重衰减（weight_decay）和梯度中心化（grad_centralization）策略，如：

```python
import mindspore as ms
from mindspore import nn
from mindvision.classification.models import resnet50

resnet = resnet50(pretrained=False)
trainable_param = resnet.trainable_params()
conv_weight, bn_weight, dense_weight = [], [], []
for _, cell in resnet.cells_and_names():
    # 判断是什么API，将对应参数加到不同列表里
    if isinstance(cell, nn.Conv2d):
        conv_weight.append(cell.weight)
    elif isinstance(cell, nn.BatchNorm2d):
        bn_weight.append(cell.gamma)
        bn_weight.append(cell.bata)
    elif isinstance(cell, nn.Dense):
        dense_weight.append(cell.weight)

other_param = []
# 所有分组里的参数不能重复，并且其交集是需要做参数更新的所有参数
for param in trainable_param:
    if (param not in conv_weight) or (param not in bn_weight) or (param not in dense_weight):
        other_param.append(param)

group_param = [{'order_params': trainable_param}]
# 每一个分组的参数列表不能是空的
if conv_weight:
    group_param.append({'params': conv_weight, 'weight_decay': 1e-4, 'lr': nn.cosine_decay_lr(0., 1e-3, cfg.total_step, cfg.step_per_epoch, cfg.decay_epoch)})
if bn_weight:
    group_param.append({'params': bn_weight, 'weight_decay': 0., 'lr': 1e-4})
if dense_weight:
    group_param.append({'params': dense_weight, 'weight_decay': 1e-5, 'lr': 1e-3})
if other_param:
    group_param.append({'params': other_param})

opt = nn.Momentum(group_param, learning_rate=1e-3, weight_decay=0.0, momentum=0.9)
```

需要注意以下几点：

1. 每一个分组的参数列表不能是空的；
2. 如果没有设置`weight_decay`和`lr`则使用优化器里设置的值，设置了的话使用分组参数字典里的值；
3. 每个分组里的`lr`都可以是静态或动态的，但不能再分组；
4. 每个分组里的`weight_decay`都需要是符合规范的浮点数；
5. 所有分组里的参数不能重复，并且其交集是需要做参数更新的所有参数。

## MindSpore的学习率衰减策略

在训练过程中，MindSpore的学习率是以参数的形式存在在网络里的，在执行优化器更新网络可训练参数前，MindSpore会调用[get_lr](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Optimizer.html#mindspore.nn.Optimizer.get_lr)
方法获取到当前step需要的学习率的值。

MindSpore的学习率支持静态、动态、分组三种，其中静态学习率在网络里是一个float32类型的Tensor。

动态学习率有两种，一种在网络里是一个长度为训练总的step数，float32类型的Tensor，如[Dynamic LR函数](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#dynamic-lr%E5%87%BD%E6%95%B0)。在优化器里有一个`global_step`的参数，每经过一次优化器更新参数会+1，MindSpore内部会根据`global_step`和`learning_rate`这两个参数来获取当前step的学习率的值；
另一种是通过构图来生成学习率的值的，如[LearningRateSchedule类](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#learningrateschedule%E7%B1%BB)。

分组学习率如上一小节参数分组中介绍的。

因为MindSpore的学习率是参数，我们也可以通过给`learning_rate`参数赋值的方式修改训练过程中学习率的值，如[LearningRateScheduler Callback](https://www.mindspore.cn/docs/zh-CN/master/_modules/mindspore/train/callback/_lr_scheduler_callback.html#LearningRateScheduler)，这种方法只支持优化器中传入静态的学习率。关键代码如下：

```python
import mindspore as ms
from mindspore import ops, nn
net = nn.Dense(1, 2)
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
print(optimizer.learning_rate.data.asnumpy())
new_lr = 0.01
# 改写learning_rate参数的值
ops.assign(optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))
print(optimizer.learning_rate.data.asnumpy())

# 0.1
# 0.01
```
