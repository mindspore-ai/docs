# 比较与tf.compat.v1.train.MomentumOptimizer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Momentum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.compat.v1.train.MomentumOptimizer

```text
tf.compat.v1.train.MomentumOptimizer(
    learning_rate,
    momentum,
    use_locking=False,
    name='Momentum',
    use_nesterov=False
) -> Tensor
```

更多内容详见[tf.compat.v1.train.MomentumOptimizer](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/MomentumOptimizer)。

## mindspore.nn.Momentum

```text
class mindspore.nn.Momentum(
    params,
    learning_rate,
    momentum,
    weight_decay=0.0,
    loss_scale=1.0,
    use_nesterov=False
)(gradients) -> Tensor
```

更多内容详见[mindspore.nn.Momentum](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Momentum.html)。

## 差异对比

TensorFlow：实现Momentum算法的优化器。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类  | TensorFlow    | MindSpore     | 差异                                                  |
| --- |-----|---------------|---------------|-----------------------------------------------------|
|参数 | 参数1 | learning_rate | learning_rate | -                                                   |
| | 参数2 | momentum      | momentum      | -                                                   |
| | 参数3 | use_locking   | -             | TensorFlow中为是否在更新操作中使用锁，默认值：False。MindSpore无此参数    |
| | 参数4 | name  | -  | 不涉及                                                 |
| | 参数5 | use_nesterov  | use_nesterov  | -                                                   |
| | 参数6 | -             | params        | 由Parameter组成的列表或字典组成的列表，TensorFlow中无此参数  |
| | 参数7 | -             | weight_decay  | 权重衰减（L2 penalty），默认值：0.0，TensorFlow中无此参数|
| | 参数8 | -             | loss_scale    | 梯度缩放系数，默认值：1.0，TensorFlow中无此参数                   |
| | 参数9 | -             |  gradients    | 参数params的梯度，TensorFlow中无此参数              |

### 代码示例

> 两API实现的功能基本一致。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

def forward_tensorflow_impl(input_np, label_np_onehot, output_channels, weight_np, bias_np, epoch, locking=False,
                            lr=0.1, momentum=0.0, use_nesterov=False, dtype=np.float32):
    tf.compat.v1.disable_eager_execution()
    input_tf = tf.constant(input_np, dtype=np.float32)
    label = tf.constant(label_np_onehot)
    if has_bias:
        net = tf.compat.v1.layers.dense(inputs=input_tf, units=output_channels, use_bias=True,
                                        kernel_initializer=tf.compat.v1.constant_initializer(
                                            weight_np.transpose(1, 0), dtype=np.float32),
                                        bias_initializer=tf.compat.v1.constant_initializer(bias_np,
                                                                                           dtype=np.float32)
                                        )
    else:
        net = tf.compat.v1.layers.dense(inputs=input_tf, units=output_channels, use_bias=False,
                                        kernel_initializer=tf.compat.v1.constant_initializer(
                                            weight_np.transpose(1, 0), dtype=np.float32)
                                        )
    criterion = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=label,
                                                          logits=net,
                                                          reduction=tf.compat.v1.losses.Reduction.MEAN)
    opt = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_locking=locking,
                                               use_nesterov=use_nesterov).minimize(criterion)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as ss:
        ss.run(init)
        num = epoch
        for _ in range(0, num):
            criterion.eval()
            ss.run(opt)
        output = net.eval()
    return output.astype(dtype)

input_np = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9]], dtype=np.float32)
weight_np = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9]], dtype=np.float32)
label_np = np.array([0, 2, 1])
label_np_onehot = np.zeros(shape=(3, 3)).astype(np.float32)
label_np_onehot[np.arange(3), label_np] = 1.0
has_bias = True
bias_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
input_channels = 3
output_channels = 3
epoch = 3
output = forward_tensorflow_impl(input_np, label_np_onehot, output_channels, weight_np, bias_np, epoch, locking=False,
                            lr=0.1, momentum=0.0, use_nesterov=False, dtype=np.float32)
print(output)
# [[0.28361297 0.5488669 0.72752017]
#  [0.4602843 1.0305372 1.4191785 ]
#  [0.6369556 1.5122076 2.1108367 ]]

# MindSpore
import numpy as np
from mindspore import Tensor
from mindspore.nn import Momentum as Momentum
from mindspore.nn import Dense, SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell, TrainOneStepCell

def forward_mindspore_impl(input_np, weight_np, label_np_onehot, has_bias, bias_np, input_channels, output_channels,
                           epoch, learning_rate=0.1, weight_decay=0.0, momentum=0.0, loss_scale=1.0, use_nesterov=False):
    input = Tensor(input_np)
    weight = Tensor(weight_np)
    label = Tensor(label_np_onehot)
    if has_bias:
        bias = Tensor(bias_np)
        net = Dense(in_channels=input_channels, out_channels=output_channels,
                    weight_init=weight, bias_init=bias, has_bias=True)
    else:
        net = Dense(in_channels=input_channels, out_channels=output_channels,
                    weight_init=weight, has_bias=False)

    criterion = SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = Momentum(params=net.trainable_params(), learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         momentum=momentum, loss_scale=loss_scale,
                         use_nesterov=use_nesterov)

    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    for _ in range(epoch):
        train_network(input, label)
    output = net(input)
    return output.asnumpy()

input_np = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9]], dtype=np.float32)
weight_np = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6],[0.7, 0.8, 0.9]], dtype=np.float32)
label_np = np.array([0, 2, 1])
label_np_onehot = np.zeros(shape=(3, 3)).astype(np.float32)
label_np_onehot[np.arange(3), label_np] = 1.0
has_bias = True
bias_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
input_channels = 3
output_channels = 3
epoch = 3
out = forward_mindspore_impl(input_np, weight_np, label_np_onehot, has_bias, bias_np, input_channels, output_channels,
                           epoch, learning_rate=0.1, weight_decay=0.0, momentum=0.0, loss_scale=1.0, use_nesterov=False)
print(out)
# [[0.28361297 0.5488669 0.72752017]
#  [0.4602843 1.0305372 1.4191784 ]
#  [0.6369556 1.5122076 2.1108367 ]]
```
