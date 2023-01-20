# Function Differences with tf.keras.layers.BatchNormalization

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/BatchNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.layers.BatchNormalization

```text
tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)(inputs, training) -> Tensor
```

For more information, see [tf.keras.layers.BatchNormalization](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/BatchNormalization).

## mindspore.nn.BatchNorm2d

```text
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BatchNorm2d.html).

## Differences

TensorFlow: Perform batch normalization on the input data.

MindSpore: Batch Normalization Layer (BNL) is applied to the input four-dimensional data. Batch normalization is applied on four-dimensional inputs (small batches of two-dimensional inputs with additional channel dimensionality) to avoid internal covariate shifts.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | x | inputs | Same function, different parameter names |
| | Parameter 2 | axis | - | The axis that should be normalized (usually the feature axis). MindSpore does not have this parameter |
| | Parameter 3 | momentum | momentum | - |
| | Parameter 4 | epsilon | eps | Same function, different parameter names |
| | Parameter 5 | center | - | If True, the offset beta is added to the normalization tensor. If False, beta is ignored. MindSpore does not have this parameter |
| | Parameter 6 | scale | - | If True, multiply by gamma. if False, do not use gamma. MindSpore does not have this parameter |
| | Parameter 7 | beta_initializer | beta_init | Same function, different parameter names |
| | Parameter 8 | gamma_initializer | gamma_init | Same function, different parameter names |
| | Parameter 9 | moving_mean_initializer | moving_mean_init | Same function, different parameter names |
| | Parameter 10 | moving_variance_initializer | moving_var_init | Same function, different parameter names |
| | Parameter 11 | beta_regularizer | - | Optional regularizer for beta weights. MindSpore does not have this parameter |
| | Parameter 12 | gamma_regularizer | - | Optional regularizer for gamma weights. MindSpore does not have this parameter |
| | Parameter 13 | beta_constraint | - | Optional constraint on the beta weight. MindSpore does not have this parameter |
| | Parameter 14 | gamma_constraint | - | Optional constraint on gamma weights. MindSpore does not have this parameter |
| | Parameter 15 | **kwargs | - | Not involved |
| | Parameter 16 | - | num_features | Number of channels, input C in Tensor shape (N,C,H,W)(N,C,H,W) |
| | Parameter 17 | - | affine | bool type. When set to True, the γ and β values can be learned. Default value: True |
| | Parameter 18 | - | use_batch_statistics | If True, the mean and variance values of the current batch data are used, and the running mean and running variance are tracked.<br /> If False, the mean and variance values of the specified values are used and no statistical values are tracked.<br /> If None, use_batch_statistics is automatically set to True or False depending on the training and validation modes. use_batch_statistics is set to True for training. use_batch_statistics is automatically set to False for validation. Default value: None |
| | Parameter 19 | - | data_format | MindSpore can specify the input data format as "NHWC" or "NCHW". Default value: "NCHW". TensorFlow does not have this parameter |
| | Parameter 20 | training | - | Not involved |

### Code Example 1

> Both APIs have the same function and are used in the same way.

```python
# TensorFlow
import tensorflow as tf

inputx = [[[[1, 2],
           [2, 1]],
          [[3, 4],
           [4, 3]]],
         [[[5, 6],
           [6, 5]],
          [[7, 8],
           [8, 7]]]]
input_tf = tf.constant(inputx, dtype=tf.float32)
output_tf = tf.keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
output = output_tf(input_tf, training=False)
print(output.numpy())
# [[[[0.999995  1.99999  ]
#    [1.99999   0.999995 ]]
#
#   [[2.999985  3.99998  ]
#    [3.99998   2.999985 ]]]
#
#
#  [[[4.999975  5.99997  ]
#    [5.99997   4.999975 ]]
#
#   [[6.9999647 7.99996  ]
#    [7.99996   6.9999647]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.BatchNorm2d(num_features=2, momentum=0.9)
input_x = Tensor(np.array([[[[1, 2], [2, 1]],
                          [[3, 4], [4, 3]]],
                          [[[5, 6], [6, 5]],
                          [[7, 8], [8, 7]]]]).astype(np.float32))
output = m(input_x)
print(output)
# [[[[0.99999493 1.9999899 ]
#    [1.9999899  0.99999493]]
#
#   [[2.9999847  3.9999797 ]
#    [3.9999797  2.9999847 ]]]
#
#
#  [[[4.9999747  5.9999695 ]
#    [5.9999695  4.9999747 ]]
#
#   [[6.9999647  7.9999595 ]
#    [7.9999595  6.9999647 ]]]]
```
