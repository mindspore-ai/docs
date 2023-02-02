# Function Differences with tf.compat.v1.train.RMSPropOptimizer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/RMSProp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.train.RMSPropOptimizer

```text
tf.compat.v1.train.RMSPropOptimizer(
    learning_rate,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
    use_locking=False,
    centered=False,
    name='RMSProp'
) -> Tensor
```

For more information, see [tf.compat.v1.train.RMSPropOptimizer](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/RMSPropOptimizer).

## mindspore.nn.RMSProp

```text
class mindspore.nn.RMSProp(
    params,
    learning_rate=0.1,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
    use_locking=False,
    centered=False,
    loss_scale=1.0,
    weight_decay=0.0
)(gradients) -> Tensor
```

For more information, see [mindspore.nn.RMSProp](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.RMSProp.html).

## Differences

TensorFlow: Implement the optimizer function of the root mean square propagation algorithm (RMSProp).

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | parameter 1 | learning_rate | learning_rate |Same function, no default values for TensorFlow |
| | parameter 2 | decay | decay | -                                                            |
| | parameter 3 | momentum | momentum |- |
| | parameter 4 | epsilon | epsilon |- |
| | parameter 5 | use_locking | use_locking |- |
| | parameter 6 | centered | centered |- |
| | parameter 7 | name | - |Not involved |
| | parameter 8 | - | params |A list composed of Parameter classes or a list composed of dictionaries, which are not available in TensorFlow. |
| | parameter 9 | - | loss_scale |Gradient scaling factor. Default value: 1.0. TensorFlow does not have this parameter |
| | parameter 10 | - | weight_decay |Weight decay (L2 penalty). Default value: 0.0. TensorFlow does not have this parameter |
| | parameter 11 | - | gradients  |Gradient of `params` in the optimizer. TensorFlow does not have this parameter |

### Code Example

> The two APIs implement the same function.

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
opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.1).minimize(criterion)
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
# [[0.94458014 0.94458014 0.94458014]
#  [0.94458014 0.94458014 0.94458014]]

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
optimizer = nn.RMSProp(params=net.trainable_params())
net_with_criterion = WithLossCell(net, criterion)
train_network = TrainOneStepCell(net_with_criterion, optimizer)
train_network.set_train()
for i in range(epoch):
    train_network(inputs, label)
out_ms = net(inputs).asnumpy()
print(out_ms)
# [[0.94458014 0.94458014 0.94458014]
#  [0.94458014 0.94458014 0.94458014]]
```
