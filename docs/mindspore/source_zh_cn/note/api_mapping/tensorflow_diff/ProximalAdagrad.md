# 比较与tf.compat.v1.train.ProximalAdagradOptimizer的功能差异

## tf.compat.v1.train.ProximalAdagradOptimizer

```text
tf.compat.v1.train.ProximalAdagradOptimizer(
    learning_rate,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    use_locking=False,
    name='ProximalAdagrad'
)(grads ) -> Tensor
```

更多内容详见 [tf.compat.v1.train.ProximalAdagradOptimizer](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/ProximalAdagradOptimizer)。

## mindspore.nn.ProximalAdagrad

```text
class mindspore.nn.ProximalAdagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    l1=0.0,
    l2=0.0,
    use_locking=False,
    loss_scale=1.0,
    weight_decay=0.0
) -> Tensor
```

更多内容详见 [mindspore.nn.ProximalAdagrad](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ProximalAdagrad.html)。

## 差异对比

TensorFlow：实现Proximal Adagrad算法的优化器功能。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，用法稍有不同。MindSpore支持参数分组`params`、梯度缩放系数`loss_scale`、权重衰减`weight_decay`等参数配置来增加相应的功能，TensorFlow无此参数功能。

| 分类 | 子类 |         TensorFlow         |   MindSpore   | 差异 |
| :-- | :-- | :-- | :-- |:--|
|参数 | 参数1 | learning_rate | learning_rate |功能一致，TensorFlow无默认值 |
| | 参数2 | initial_accumulator_value | accum |功能一致，参数名不同 |
| | 参数3 | l1_regularization_strength | l1 |功能一致，参数名不同 |
| | 参数4 | l2_regularization_strength | l2 |功能一致，参数名不同 |
| | 参数5 | use_locking | use_locking |- |
| | 参数6 | name | - |不涉及 |
| | 参数7 | - | params |MindSpore提供参数分组功能，且支持为不同参数组设置不同配置值，通过入参`params`传入参数组字典实现，TensorFlow没有此入参配置 |
| | 参数8 | - | loss_scale |梯度缩放系数，TensorFlow无此参数 |
| | 参数9 | - | weight_decay |实现对需要优化的参数使用权重衰减的策略，以避免模型过拟合问题，TensorFlow无此参数 |
| | 参数10 | - | grads  |反向输入，TensorFlow无此参数 |

### 代码示例

> 实现功能一致。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

param_np = np.ones(7).astype(np.float32)
indices = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)
label = np.zeros((2, 3)).astype(np.float32)
label_shape = (2, 3)
axis = 0
epoch = 3
param_tf = tf.Variable(param_np)
indices_tf = tf.Variable(indices)
label = tf.Variable(label)
net = tf.raw_ops.GatherV2(params=param_tf, indices=indices_tf, axis=axis, batch_dims=0, name=None)
net = tf.reshape(net, label_shape)
criterion = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=label, logits=net,
                                                      reduction=tf.compat.v1.losses.Reduction.MEAN)
opt = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=0.001).minimize(criterion)
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as ss:
    ss.run(init)
    for i in range(epoch):
        loss = criterion.eval()
        ss.run(opt)
    output = net.eval()
    net = ss.run(net)
out_tf = output.astype(np.float32)
print(out_tf)
# [[0.9987219 0.9987219 0.9987219]
#  [0.9987219 0.9987219 0.9987219]]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as op
from mindspore import Parameter
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn import Cell

class NetWithGatherV2(Cell):
    def __init__(self, param_np, label, axis=0):
        super(NetWithGatherV2, self).__init__()
        self.param = Parameter(Tensor(param_np), name="w1")
        self.gatherv2 = op.GatherV2()
        self.reshape = op.Reshape()
        self.axis = axis
        self.label = label

    def construct(self, indices):
        x = self.gatherv2(self.param, indices, self.axis)
        return self.reshape(x, self.label)


param_np = np.ones(7).astype(np.float32)
indices = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)
label = np.zeros((2, 3)).astype(np.float32)
label_shape = (2, 3)
epoch = 3
inputs = Tensor(indices)
label = Tensor(label)
net = NetWithGatherV2(param_np, label_shape, axis=0)
criterion = SoftmaxCrossEntropyWithLogits(reduction='mean')
optimizer = nn.ProximalAdagrad(params=net.trainable_params())
net_with_criterion = WithLossCell(net, criterion)
train_network = TrainOneStepCell(net_with_criterion, optimizer)
train_network.set_train()
for i in range(epoch):
    train_network(inputs, label)
out_ms = net(inputs).asnumpy()
print(out_ms)
# [[0.9987219 0.9987219 0.9987219]
#  [0.9987219 0.9987219 0.9987219]]
```
