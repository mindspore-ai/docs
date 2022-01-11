# 比较与tf.keras.initializers.VarianceScaling的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/initXavierUniform.md " target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## tf.keras.initializers.VarianceScaling

```python
tf.keras.initializers.VarianceScaling(
    scale=1.0, mode='fan_in', distribution='truncated_normal',
    seed=None
)
```

更多内容详见[tf.keras.initializers.VarianceScaling](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling)。

## mindspore.common.initializer.XavierUniform

```python
mindspore.common.initializer.XavierUniform(gain=1)
```

更多内容详见[mindspore.common.initializer.XavierUniform](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.common.initializer.html#mindspore.common.initializer.XavierUniform)。

## 使用方式

Xavier初始化方法主要思想是，为使得网络中信息更好的流动，每一层输出的方差应该尽量相等，基于此，经过部分数学公式推导，Xavier均匀分布服从U(−bound, bound)，boud的计算方式为gain * sqrt ( 6 / (fan_in+fan_out))，其中gain通常根据激活函数决定，fan_in和fan_out分别代表输入单元个数和输出单元个数。

- MindSpore：实现方式与上述计算方式相同，其中gain值通过函数入参传入，默认值为1.0。

- TensorFlow：主要有三个入参需要注意：

  `distribution`：可设置为"truncated_normal"、"untruncated_normal"、"uniform"，代表生成的数据分布；

  `mode`：可设置为"fan_in"、"fan_out"、"fan_avg"，分别表示分母的取值为fan_in、fan_out、(fan_in + fan_out) / 2；

  `scale`：为公式中的gain，默认值：1.0。

  当入参配置为`distribution`="uniform"，`mode`="fan_avg"时，生成数据的方式与`mindspore.common.initializer.XavierUniform`相同。

## 代码示例

> 以下代码结果具有随机性。

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
