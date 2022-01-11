# 比较与tf.keras.initializers.RandomUniform的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/initUniform.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## tf.keras.initializers.RandomUniform

```python
tf.keras.initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None, dtype=tf.dtypes.float32
)
```

更多内容详见[tf.keras.initializers.RandomUniform](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/initializers/RandomUniform)。

## mindspore.common.initializer.Uniform

```python
class mindspore.common.initializer.Uniform(scale=0.07)
```

更多内容详见[mindspore.common.initializer.Uniform](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.common.initializer.html?#mindspore.common.initializer.Uniform)。

## 使用方式

TensorFlow：通过入参`minval`和`maxval`分别指定均匀分布的上下界，即U(-minval, maxval)。默认值：minval=-0.05, maxval=0.05。

MindSpore：仅通过一个入参`scale`指定均匀分布的范围，即U(-scale, scale)。默认值：scale=0.7。

## 代码示例

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
