# Function Differences with tf.keras.initializers.RandomNormal

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initNormal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.initializers.RandomNormal

```python
tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.05, seed=None, dtype=tf.dtypes.float32
)
```

For more information, see [tf.keras.initializers.RandomNormal](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/initializers/RandomNormal).

## mindspore.common.initializer.Normal

```python
mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)
```

For more information, see [mindspore.common.initializer.Normal](https://mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal).

## Usage

TensorFlow: generate a normal distribution with a mean of 0.0 and a standard deviation of 0.05 by default. Default values: mean=0.0, stddev=0.05.

MindSpore: generate a normal distribution with a mean of 0.0 and a standard deviation of 0.01 by default. Default values: mean=0.0, sigma=0.01.

## Code Example

> The following results are random.

```python
import tensorflow as tf

init = tf.keras.initializers.RandomNormal()

x = init(shape=(2, 2))

with tf.Session() as sess:
    print(x.eval())

# out:
# [[-1.4192176  1.9695756]
#  [ 1.6240929  0.9677597]]
```

```python
import mindspore as ms
from mindspore.common.initializer import Normal, initializer

x = initializer(Normal(), shape=[2, 2], dtype=ms.float32)
print(x)

# out:
# [[ 0.01005767 -0.00049193]
#  [-0.00026987  0.02598832]]
```
