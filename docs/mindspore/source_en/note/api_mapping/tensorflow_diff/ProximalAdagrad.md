# Function Differences with tf.compat.v1.train.ProximalAdagradOptimizer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/ProximalAdagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.train.ProximalAdagradOptimizer

```text
tf.compat.v1.train.ProximalAdagradOptimizer(
    learning_rate,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    use_locking=False,
    name='ProximalAdagrad'
) -> Tensor
```

For more information, see [tf.compat.v1.train.ProximalAdagradOptimizer](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/ProximalAdagradOptimizer).

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
)(grads) -> Tensor
```

For more information, see [mindspore.nn.ProximalAdagrad](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ProximalAdagrad.html).

## Differences

TensorFlow: Implement the optimizer function of the Proximal Adagrad algorithm.

MindSpore: MindSpore API basically implements the same function as TensorFlow, with slightly different usage. MindSpore supports parameter grouping `params`, gradient scaling factor `loss_scale`, weight decay `weight_decay` and other parameter configurations to add the corresponding functions, and TensorFlow does not have this parameter function.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| :-- | :-- | :-- | :-- |:--|
|Parameters | parameter 1 | learning_rate | learning_rate |Same function, no default values for TensorFlow |
| | parameter 2 | initial_accumulator_value | accum |Same function, different parameter names |
| | parameter 3 | l1_regularization_strength | l1 |Same function, different parameter names |
| | parameter 4 | l2_regularization_strength | l2 |Same function, different parameter names |
| | parameter 5 | use_locking | use_locking |- |
| | parameter 6 | name | - |Not involved |
| | parameter 7 | - | params |MindSpore provides parameter grouping, and supports setting different configuration values for different parameter groups, which is implemented by passing the parameter group dictionary through the parameter `params`, which is not available in TensorFlow. |
| | parameter 8 | - | loss_scale |Gradient scaling factor. TensorFlow does not have this parameter |
| | parameter 9 | - | weight_decay |Implement a strategy of using weight decay for parameters that need to be optimized to avoid model overfitting problems. TensorFlow does not have this parameter |
| | parameter 10 | - | grads  |Reverse input. TensorFlow does not have this parameter |

### Code Example

> The two APIs achieve functional consistency.

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

