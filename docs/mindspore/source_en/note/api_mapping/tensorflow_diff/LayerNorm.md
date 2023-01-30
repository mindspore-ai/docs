# Function Differences with tf.keras.layers.LayerNormalization

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/LayerNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.layers.LayerNormalization

```python
class tf.keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)(x) -> Tensor
```

For more information, see [tf.keras.layers.LayerNormalization](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/LayerNormalization).

## mindspore.nn.LayerNorm

```python
class mindspore.nn.LayerNorm(
    normalized_shape,
    begin_norm_axis=-1,
    begin_params_axis=-1,
    gamma_init='ones',
    beta_init='zeros',
    epsilon=1e-7
)(x) -> Tensor
```

For more information, see [mindspore.nn.LayerNorm](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LayerNorm.html).

## Differences

TensorFlow: Apply Layer Normalization to the mini-batch input, where the parameters `center` and `scale` control whether to use beta and gamma, `beta_regularizer` and `gamma_regularizer` are used to control whether to use optional regularizers for beta and gamma, and the parameters `beta_constraint` and `gamma_constraint` are used to control whether to use optional constraints for beta and gamma.

MindSpore: When all the parameters in this API of TensorFlow are default, MindSpore API basically implements the same function as TensorFlow. However, the parameters `center` and `scale` do not exist in MindSpore, so the function of ignoring beta and gamma cannot be implemented; the parameters `beta_regularizer`, `gamma_regularizer`, `beta_constraint`, and `gamma_constraint` do not exist, so the corresponding functions are not supported. Also MindSpore adds the parameter `begin_params_axis` to control the dimensionality of the first parameter (beta, gamma) and the parameter `normalized_shape` to control the specific dimensionality of the mean and standard deviation calculation.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | axis | begin_norm_axis | Same function, different parameter names|
| | Parameter 2 | epsilon | epsilon | Same function, same parameter name, different default value |
| | Parameter 3 | center | - | This parameter is used in TensorFlow to control whether to use the beta offset. MindSpore does not have this parameter|
| | Parameter 4 | scale | - | This parameter is used in TensorFlow to control whether gamma is used. MindSpore does not have this parameter|
| | Parameter 5 | beta_initializer | beta_init | Same function, different parameter names|
| | Parameter 6 | gamma_initializer | gamma_init | Same function, different parameter names|
| | Parameter 7 | beta_regularizer | - | This parameter is used in TensorFlow to control whether an optional regularizer with beta weights is used. MindSpore does not have this parameter|
| | Parameter 8 | gamma_regularizer | - | This parameter is used in TensorFlow to control whether the optional regularizer with gamma weights is used. MindSpore does not have this parameter|
| | Parameter 9 | beta_constraint | - | This parameter is used in TensorFlow to control whether the optional constraint of beta weights is used. MindSpore does not have this parameter|
| | Parameter 10 | gamma_constraint | - | This parameter is used in TensorFlow to control whether the optional constraint of gamma weights is used. MindSpore does not have this parameter|
| | Parameter 11 | **kwargs | - | Not involved|
| | Parameter 12 | x | x | - |
| | Parameter 13 | - | normalized_shape | This parameter is not available in TensorFlow, and this parameter in MindSpore controls the specific dimension of the mean and standard deviation calculation|
| | Parameter 14 | - | begin_params_axis | No such parameter in TensorFlow. This parameter in MindSpore determines the dimension of the first parameter (beta, gamma) and is used to broadcast the built-in parameters scale and centering |

### Code Example

> When all parameters in this API of TensorFlow are default, MindSpore and TensorFlow achieve basically the same function.

```python
# TensorFlow
import tensorflow as tf

inputs = tf.ones([20, 5, 10, 10])
layer = tf.keras.layers.LayerNormalization(axis=-1)
output = layer(inputs)
print(output.shape)
# (20, 5, 10, 10)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.nn as nn

x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
shape1 = x.shape[1:]
m = nn.LayerNorm(shape1, begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)
# (20, 5, 10, 10)
```
