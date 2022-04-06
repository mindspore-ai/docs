# 比较与tf.keras.initializers.Constant的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/initConst.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.initializers.Constant

```python
tf.keras.initializers.Constant(value=0)
```

更多内容详见[tf.keras.initializers.Constant](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Constant)。

## mindspore.common.initializer.Constant

```python
mindspore.common.initializer.Constant(value)
```

更多内容详见[mindspore.common.initializer.Constant](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant)。

## 使用方式

TensorFlow: 函数入参`value`支持标量，列表，元组，数组类型。假设需要创建一个指定shape的张量，且此接口的入参`value`类型为列表或数组时，`value`包含的元素数量必须小于等于指定shape的元素数量，小于的情况下，`value`的最后一个元素值用来填充剩余的位置。

MindSpore：函数入参`value`支持标量和数组类型。`value`为数组时，只能生成与`value`形状相同的张量。

## 代码示例

以输入为数组为例，代码样例如下：

TensorFlow:

```python
import numpy as np
import tensorflow as tf

value = np.array([0, 1, 2, 3, 4, 5, 6, 7])
value = value.reshape([2, 4])

init = tf.keras.initializers.Constant(value)

x = init(shape=(2, 4))
y = init(shape=(3, 4))

with tf.Session() as sess:
    print(x.eval(), "\n")
    print(y.eval())

# out:
# [[0. 1. 2. 3.]
#  [4. 5. 6. 7.]]

# [[0. 1. 2. 3.]
#  [4. 5. 6. 7.]
#  [7. 7. 7. 7.]]
```

MindSpore:

```python
import numpy as np
import mindspore as ms
from mindspore.common.initializer import initializer, Constant

value = np.array([0, 1, 2, 3, 4, 5, 6, 7])
value = value.reshape([2, 4])

x = initializer(Constant(value), shape=[2, 4], dtype=ms.float32)

print(x)

# out:
# [[0. 1. 2. 3.]
#  [4. 5. 6. 7.]]
```
