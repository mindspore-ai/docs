# LossScale

<!-- TOC -->

- [LossScale](#LossScale)
    - [概述](#概述)
    - [FixedLossScaleManager](#FixedLossScaleManager)
    - [DynamicLossScaleManager](#DynamicLossScaleManager)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/lossscale.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

在混合精度中，会使用float16类型来替代float32类型存储数据，从而达到减少内存和提高计算速度的效果。但是由于float16类型要比float32类型表示的范围小很多，所以当某些参数（比如说梯度）在训练过程中变得很小时，就会发生数据下溢的情况。而LossScale正是为了解决float16类型数据下溢问题的，LossScale的主要思想是在计算loss时，将loss扩大一定的倍数，由于链式法则的存在，梯度也会相应扩大，然后在优化器更新权重时再缩小相应的倍数，从而避免了数据下溢的情况又不影响计算结果。

MindSpore中提供了两种LossScale的方式，分别是`FixedLossScaleManager`和`DynamicLossScaleManager`，一般需要和Model配合使用。

## FixedLossScaleManager

`FixedLossScaleManager`在进行scale的时候，不会改变scale的大小，scale的值由入参loss_scale控制，可以由用户指定，不指定则取默认值。`FixedLossScaleManager`的另一个参数是`drop_overflow_update`，用来控制发生溢出时是否更新参数。一般情况下LossScale功能不需要和优化器配合使用，但使用`FixedLossScaleManager`时，如果`drop_overflow_update`为False，那么优化器需设置`loss_scale`的值，且`loss_scale`的值要与`FixedLossScaleManager`的相同。

`FixedLossScaleManager`具体用法如下：

```python
from mindspore import Model, nn, FixedLossScaleManager

net = Net()
#1) Drop the parameter update if there is an overflow
loss_scale_manager = FixedLossScaleManager()
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)

#2) Execute parameter update even if overflow occurs
loss_scale = 1024.0
loss_scale_manager = FixedLossScaleManager(loss_scale, False)
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9, loss_scale=loss_scale)
model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
```

## DynamicLossScaleManager

`DynamicLossScaleManager`在训练过程中可以动态改变scale的大小，在没有发生溢出的情况下，要尽可能保持较大的scale。`DynamicLossScaleManager`会首先将scale设置为一个初始值，该值由入参init_loss_scale控制。在训练过程中，如果不发生溢出，在更新scale_window次参数后，会尝试扩大scale的值，如果发生了溢出，则跳过参数更新，并缩小scale的值，入参scale_factor是控制扩大或缩小的步数，scale_window控制没有发生溢出时，最大的连续更新步数。

`DynamicLossScaleManager`的具体用法如下：

```python
from mindspore import Model, nn, DynamicLossScaleManager

net = Net()
scale_factor = 4
scale_window = 3000
loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)
```