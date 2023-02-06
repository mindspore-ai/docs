# Function Differences with tf.keras.optimizers.Adam

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Adam.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.optimizers.Adam

```text
tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
    **kwargs
) -> Tensor
```

For more information, see [tf.keras.optimizers.Adam](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/Adam).

## mindspore.nn.Adam

```text
class mindspore.nn.Adam(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    use_locking=False,
    use_nesterov=False,
    weight_decay=0.0,
    loss_scale=1.0,
    use_amsgrad=False,
    **kwargs
)(gradients) -> Tensor
```

For more information, see [mindspore.nn.Adam](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Adam.html).

## Differences

TensorFlow: Implement the optimizer function of Adam algorithm.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1  | learning_rate | learning_rate | -                                                                                                          |
|      | Parameter 2  | beta_1        | beta1         | Same function, different parameter names                                                                                      |
|      | Parameter 3  | beta_2        | beta2         | Same function, different parameter names                                                                                      |
|      | Parameter 4  | epsilon       | eps           | Same function, different parameter names and default values                         |
|      | Parameter 5  | amsgrad       | use_amsgrad   | Same function, different parameter names                                                                                      |
|      | Parameter 6 | name                   | -             | Not involved |
|      | Parameter 7 |  **kwargs                   | **kwargs       | Not involved |
|      | Parameter 8  | -             | params        | A list of Parameter classes or a list of dictionaries. TensorFlow does not have this parameter.               |
|      | Parameter 9  | -             | use_locking   | Based on this parameter, MindSpore can decide whether to lock protection for parameter updates. TensorFlow does not have this parameter.                                   |
|      | Parameter 10  | -             | use_nesterov  | Based on this parameter, MindSpore can decide whether to update the gradient using the Nesterov Accelerated Gradient (NAG) algorithm. TensorFlow does not have this parameter.  |
|      | Parameter 11  | -             | weight_decay  | weight decay (L2 penalty), default value: 0.0. TensorFlow does not have this parameter.         |
|      | Parameter 12 | -             | loss_scale    | gradient scaling factor, default value: 1.0. TensorFlow does not have this parameter.        |
| Input | Single input | -             | gradients    | The gradient of params. TensorFlow does not have this parameter.                                      |

### Code Example

> The two APIs achieve the same function.

```python
# TensorFlow
import tensorflow as tf
import numpy as np
input_n = 2
output_c = 2
input_channels = 2
output_channels = 2
dtype = np.float32
lr = 0.001
epoch = 100
initial_accumulator_value = 0.1
eps = 1e-7
input_np = np.array([[1, 2], [3, 4]]).astype(dtype)
weight_np = np.array([[1, 2], [3, 4]]).astype(dtype)
bias_np = np.array([0.5, 0.5]).astype(dtype)
label_np = np.array([1,0]).astype(int)
label_np_onehot = np.zeros(shape=(input_n, output_c)).astype(dtype)
label_np_onehot[np.arange(input_n), label_np] = 1.0
tf.compat.v1.disable_eager_execution()
input_tf = tf.constant(input_np, dtype=np.float32)
label = tf.constant(label_np_onehot)
net = tf.compat.v1.layers.dense(
    inputs=input_tf,
    units=output_channels,
    use_bias=True,
    kernel_initializer=tf.compat.v1.constant_initializer(
        weight_np.transpose(1, 0),
        dtype=np.float32
    ),
    bias_initializer=tf.compat.v1.constant_initializer(bias_np,dtype=np.float32)
)
criterion = tf.compat.v1.losses.softmax_cross_entropy(
    onehot_labels=label,
    logits=net,
    reduction=tf.compat.v1.losses.Reduction.MEAN
)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8).minimize(criterion)
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as ss:
    ss.run(init)
    num = epoch
    for _ in range(0, num):
        criterion.eval()
        ss.run(opt)
    output = net.eval()
print(output.astype(dtype))
# [[ 5.898781 11.101219]
#  [12.297218 24.702782]]

# MindSpore
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from mindspore.nn import Adam
import numpy as np
input_n = 2
output_c = 2
input_channels = 2
output_channels = 2
dtype = np.float32
lr = 0.001
epoch = 100
accum = 0.1
loss_scale = 1.0
weight_decay = 0
input_np = np.array([[1, 2], [3, 4]]).astype(dtype)
weight_np = np.array([[1, 2], [3, 4]]).astype(dtype)
bias_np = np.array([0.5, 0.5]).astype(dtype)
label_np = np.array([1, 0]).astype(int)
label_np_onehot = np.zeros(shape=(input_n, output_c)).astype(dtype)
label_np_onehot[np.arange(input_n), label_np] = 1.0
input_me = Tensor(input_np.copy())
weight = Tensor(weight_np.copy())
label = Tensor(label_np_onehot.copy())
bias = Tensor(bias_np.copy())
net = Dense(
    in_channels=input_channels,
    out_channels=output_channels,
    weight_init=weight,
    bias_init=bias,
    has_bias=True
)
criterion = SoftmaxCrossEntropyWithLogits(reduction='mean')
optimizer = Adam(params=net.trainable_params(), eps=1e-8, learning_rate=lr)
net_with_criterion = WithLossCell(net, criterion)
train_network = TrainOneStepCell(net_with_criterion, optimizer)
train_network.set_train()
num = epoch
for _ in range(0, num):
    train_network(input_me, label)
output = net(input_me)
print(output.asnumpy())
# [[ 5.8998876 11.100113 ]
#  [12.299808  24.700195 ]]
```
