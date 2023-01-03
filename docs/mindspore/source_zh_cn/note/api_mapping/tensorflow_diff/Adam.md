# 比较与tf.keras.optimizers.Adam的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Adam.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.keras.optimizers.Adam](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/Adam)。

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
    use_amsgrad=False
)(gradients) -> Tensor
```

更多内容详见[mindspore.nn.Adam](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Adam.html)。

## 差异对比

TensorFlow：对所有参数采用Adam方法进行梯度下降。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，MindSpore的API还可以根据params里的设置，对不同参数采用不同的学习率进行更新。

| 分类 | 子类   | TensorFlow    | MindSpore     | 差异                                                                                                       |
| ---- | ------ | ------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| 参数 | 参数1  | learning_rate | learning_rate | -                                                                                                          |
|      | 参数2  | beta_1        | beta1         | 功能一致，参数名不同                                                                                      |
|      | 参数3  | beta_2        | beta2         | 功能一致，参数名不同                                                                                      |
|      | 参数4  | epsilon       | eps           | 功能一致，参数名不同，但是TensorFlow上的默认值是1e-8，而MindSpore上的默认值是1e-7                         |
|      | 参数5  | amsgrad       | use_amsgrad   | 功能一致，参数名不同                                                                                      |
|      | 参数6 | name                   | -             | 不涉及 |
|      | 参数7 |  **kwargs                   | -             | 不涉及 |
|      | 参数8  | -             | params        | MindSpore中可以根据该参数，可以给不同参数设置不同的学习率、权重衰减值等。TensorFlow无此参数                |
|      | 参数9  | -             | use_locking   | MindSpore中可以根据该参数，决定是否对参数更新加锁保护。TensorFlow无此参数                                   |
|      | 参数10  | -             | use_nesterov  | MindSpore中可以根据该参数，决定是否使用Nesterov Accelerated Gradient(NAG)算法更新梯度。TensorFlow无此参数  |
|      | 参数11  | -             | weight_decay  | MindSpore中可以根据该参数，设定权重衰减的值。TensorFlow无此参数                                            |
|      | 参数12 | -             | loss_scale    | MindSpore中可以根据该参数，设定梯度缩放系数。TensorFlow无此参数                                             |
|      | 参数13 | -             | gradients    | MindSpore中的输入梯度，TensorFlow无此参数                                       |

### 代码示例

> 实现功能一致。

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
# [[ 5.8998876 11.100113 ]
#  [12.299808  24.700195 ]]

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
