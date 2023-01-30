# Function Differences with tf.keras.initializers.TruncatedNormal

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initTruncNormal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.initializers.TruncatedNormal

```python
tf.keras.initializers.TruncatedNormal(
    mean=0.0, stddev=0.05, seed=None, dtype=tf.dtypes.float32
)
```

For more information, see [tf.keras.initializers.TruncatedNormal](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/initializers/TruncatedNormal).

## mindspore.common.initializer.TruncatedNormal

```python
mindspore.common.initializer.TruncatedNormal(sigma=0.01)
```

For more information, see [mindspore.common.initializer.TruncatedNormal](https://mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.TruncatedNormal).

## Usage

TensorFlow: By default, based on a normal distribution with mean 0.0 and standard deviation 0.05, limit the variables to be within 2 times the standard deviation from the mean and regenerate the distribution. Default values: mean=0.0, stddev=0.05.

MindSpore: By default, based on a normal distribution with mean 0.0 and standard deviation 0.01, limit the variables to be within 2 times the standard deviation from the mean and regenerate the distribution. Default values: sigma=0.01.

## Code Example

> The following results are random.

```python
import tensorflow as tf

init = tf.keras.initializers.TruncatedNormal()
x = init(shape=(1, 2))

with tf.Session() as sess:
    print(x.eval())

# out:
# [[-0.71518797 -0.6879003 ]]
```

```python
import mindspore as ms
from mindspore.common.initializer import TruncatedNormal, initializer

x = initializer(TruncatedNormal(), shape=[1, 2], dtype=ms.float32)
print(x)

# out:
# [[0.01012452 0.00313655]]
```
