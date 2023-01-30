# Function Differences with tf.keras.initializers.VarianceScaling

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initXavierUniform.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.initializers.VarianceScaling

```python
tf.keras.initializers.VarianceScaling(
    scale=1.0, mode='fan_in', distribution='truncated_normal',
    seed=None
)
```

For more information, see [tf.keras.initializers.VarianceScaling](https://tensorflow.google.cn/api_docs/python/tf/keras/initializers/VarianceScaling).

## mindspore.common.initializer.XavierUniform

```python
mindspore.common.initializer.XavierUniform(gain=1)
```

For more information, see [mindspore.common.initializer.XavierUniform](https://mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.XavierUniform).

## Usage

The main idea of the Xavier initialization method is that the variance of the output of each layer should be as equal as possible in order to make the information flow better in the network. Based on this, after partial derivation of the mathematical formula, the Xavier uniform distribution obeys U(-bound, bound), and the boud is calculated as gain * sqrt ( 6 / (fan_in+fan_out)), where gain is usually determined according to the activation function, and fan_in and fan_out represent the number of input units and the number of output units, respectively.

- MindSpore: The implementation is the same as the above calculation, where the gain value is passed in through the function entry and the default value is 1.0.

- TensorFlow: There are three main input parameters to note:

  `distribution`: can be set to "truncated_normal", "untruncated_normal", "uniform", representing the generated data distribution.

  `mode`: can be set to "fan_in", "fan_out", "fan_avg", indicating that the denominators take the values of fan_in, fan_out, (fan_in + fan_out) / 2, respectively.

  `scale`: the gain in the formula. Default value: 1.0.

  When the input is configured as `distribution`="uniform" and `mode`="fan_avg", the data is generated in the same way as `mindspore.common.initializer.XavierUniform`.

## Code Example

> The following results are random.

```python
import tensorflow as tf

init = tf.keras.initializers.VarianceScaling(mode="fan_avg", distribution="uniform")
x = init(shape=(1, 2))

with tf.Session() as sess:
    print(x.eval())

# Out：
# [[-1.1756071   0.11235881]]
```

```python
import mindspore as ms
from mindspore.common.initializer import XavierUniform, initializer

x = initializer(XavierUniform(), shape=[1, 2], dtype=ms.float32)
print(x)

# out:
# [[-0.86959594  1.2498646 ]]
```
