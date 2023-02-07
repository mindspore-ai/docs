# Function Differences with tf.compat.v1.layers.Dense

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Dense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.layers.Dense

```text
class tf.compat.v1.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    **kwargs
)(x) -> Tensor
```

For more information, see [tf.compat.v1.layers.Dense](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/layers/Dense).

## mindspore.nn.Dense

```text
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init='normal',
    bias_init='zeros',
    has_bias=True,
    activation=None
)(x) -> Tensor
```

For more information, see [mindspore.nn.Dense](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Dense.html).

## Differences

TensorFlow: Fully connected layer that implements the matrix multiplication operation.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1  | units                | out_channels | Same function, different parameter names           |
|      | Parameter 2  | activation           | activation   | -   |
|      | Parameter 3  | use_bias             | has_bias     | Same function, different parameter names                |
|      | Parameter 4  | kernel_initializer   | weight_init  | Same function, different parameter names                |
|      | Parameter 5  | bias_initializer     | bias_init    | Same function, different parameter names            |
|      | Parameter 6  | kernel_regularizer   | -    | The regular function of the weight matrix. MindSpore does not have this parameter.        |
|      | Parameter 7  | bias_regularizer     |    -     | The regularization function for the deviation. MindSpore does not have this parameter.               |
|      | Parameter 8  | activity_regularizer |    -          | The regularization function for the output. MindSpore does not have this parameter.          |
|      | Parameter 9  | kernel_constraint    |    -   | Optional projection functions that will be applied to the kernel after the `Optimizer` program is updated (e.g., for implementing norm constraints or value constraints on layer weights). The function must take as input the unprojected variables and must return the projected variables (which must have the same shape). It is not safe to use constraints when doing asynchronous distributed training. MindSpore does not have this parameter |
|      | Parameter 10 | bias_constraint      |     -   | Optional projection function to be applied to the deviation after being updated by `Optimizer`. MindSpore does not have this parameter |
|      | Parameter 11 | trainable            |     -         | Boolean. If `True`, also adds the variable to the graph collection `GraphKeys.TRAINABLE_VARIABLES`. MindSpore does not have this parameter. |
|      | Parameter 12 | name     |     -     | Not involved   |
|      | Parameter 13 | **kwargs   |     -    | Not involved    |
|      | Parameter 14 | -    |     in_channels         | The spatial dimension of the input. TensorFlow does not have this parameter    |
|  Input   | Single input | x                 |     x         | -    |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf
from tensorflow.compat.v1 import layers
import numpy as np

model = layers.Dense(4)
x = tf.constant(np.array([[180, 234, 154], [244, 48, 247]]),dtype=tf.float32)
output = model(x)
print(output.shape)
# (2, 4)

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(x)
print(output.shape)
# (2, 4)
```
