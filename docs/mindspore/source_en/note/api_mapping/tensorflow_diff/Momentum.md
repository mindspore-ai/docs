# Function Differences with tf.compat.v1.train.MomentumOptimizer

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Momentum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.compat.v1.train.MomentumOptimizer](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/MomentumOptimizer).

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

For more information, see [mindspore.nn.Momentum](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Momentum.html).

## Differences

TensorFlow: Optimizer that implements the Momentum algorithm.

MindSpore: MindSpore API basically implements the same functions as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter1 | learning_rate | learning_rate | -  |
| | Parameter2 | momentum      | momentum      | -    |
| | Parameter3 | use_locking   | -             | In TensorFlow, whether to use locks in update operations. Default value: False. MindSpore does not have this parameter |
| | Parameter4 | name  | -  | Not involved     |
| | Parameter5 | use_nesterov  | use_nesterov  | -           |
| | Parameter6 | -             | params        | A list consisting of a Parameter or a dictionary, which is not available in TensorFlow  |
| | Parameter7 | -             | weight_decay  | Weight decay (L2 penalty), default value: 0.0. No parameter in TensorFlow |
| | Parameter8 | -             | loss_scale    | Gradient scaling factor, default value: 0.0. No parameter in TensorFlow |
| | Parameter9 | -             |  gradients    | The gradient of the parameter params, no parameter in TensorFlow              |

### Code Example

> The two APIs basically achieve the same function.

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
