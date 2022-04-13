# 比较与tf.keras.initializers.TruncatedNormal 的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/initTruncNormal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.initializers.TruncatedNormal

```python
tf.keras.initializers.TruncatedNormal(
    mean=0.0, stddev=0.05, seed=None, dtype=tf.dtypes.float32
)
```

更多内容详见[tf.keras.initializers.TruncatedNormal](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/initializers/TruncatedNormal)。

## mindspore.common.initializer.TruncatedNormal

```python
mindspore.common.initializer.TruncatedNormal(sigma=0.01)
```

更多内容详见[mindspore.common.initializer.TruncatedNormal](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.TruncatedNormal)。

## 使用方式

TensorFlow: 默认在均值为0.0，标准差为0.05的正态分布的基础上，限制变量与均值的差值在2倍标准差范围内，并重新生成分布。默认值：mean=0.0, stddev=0.05。

MindSpore：默认在均值为0.0，标准差为0.01的正态分布的基础上，限制变量与均值的差值在2倍标准差范围内，并重新生成分布。默认值：sigma=0.01。

## 代码示例

> 以下结果具有随机性。

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
