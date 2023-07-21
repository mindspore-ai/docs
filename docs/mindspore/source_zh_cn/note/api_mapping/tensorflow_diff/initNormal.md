# 比较与tf.keras.initializers.RandomNormal的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/initNormal.md )

## tf.keras.initializers.RandomNormal

```python
tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.05, seed=None, dtype=tf.dtypes.float32
)
```

更多内容详见[tf.keras.initializers.RandomNormal](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/initializers/RandomNormal)。

## mindspore.common.initializer.Normal

```python
mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)
```

更多内容详见[mindspore.common.initializer.Normal](https://mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal)。

## 使用方式

TensorFlow：默认生成均值为0.0，标准差为0.05的正态分布。默认值：mean=0.0，stddev=0.05。

MindSpore：默认生成均值为0.0，标准差为0.01的正态分布。默认值：sigma=0.01，mean=0.0。

## 代码示例

> 以下结果具有随机性。

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
