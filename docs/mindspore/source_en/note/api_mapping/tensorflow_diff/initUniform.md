# Function Differences with tf.keras.initializers.RandomUniform

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initUniform.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.initializers.RandomUniform

```python
tf.keras.initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None, dtype=tf.dtypes.float32
)
```

For more information, see [tf.keras.initializers.RandomUniform](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/initializers/RandomUniform).

## mindspore.common.initializer.Uniform

```python
class mindspore.common.initializer.Uniform(scale=0.07)
```

For more information, see [mindspore.common.initializer.Uniform](https://mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html?#mindspore.common.initializer.Uniform).

## Usage

TensorFlow: The upper and lower bounds of the uniform distribution are specified by the entry `minval` and `maxval`, i.e., U(-minval, maxval), respectively. Default values: minval=-0.05, maxval=0.05.

MindSpore: The range of the uniform distribution is specified by only one input `scale`, i.e. U(-scale, scale). Default value: scale=0.7.

## Code Example

```python
import tensorflow as tf

init = tf.keras.initializers.RandomUniform()
x = init(shape=(1, 2))

with tf.Session() as sess:
    print(x.eval())

# Out：
# [[0.9943197  0.93056154]]
```

```python
import mindspore as ms
from mindspore.common.initializer import Uniform, initializer

x = initializer(Uniform(), shape=[1, 2], dtype=ms.float32)
print(x)

# out:
# [[0.01140347 0.0076657 ]]
```

