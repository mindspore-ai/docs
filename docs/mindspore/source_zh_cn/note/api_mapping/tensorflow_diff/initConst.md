# 比较与tf.keras.initializers.Constant的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/initConst.md)

## tf.keras.initializers.Constant

```python
tf.keras.initializers.Constant(value=0)
```

更多内容详见[tf.keras.initializers.Constant](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/initializers/Constant)。

## mindspore.common.initializer.Constant

```python
mindspore.common.initializer.Constant(value)
```

更多内容详见[mindspore.common.initializer.Constant](https://mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant)。

## 使用方式

TensorFlow：函数入参`value`支持标量，列表，元组，数组类型。假设需要创建一个指定shape的张量，且此接口的入参`value`类型为列表或数组时，`value`包含的元素数量必须小于等于指定shape的元素数量，小于的情况下，`value`的最后一个元素值用来填充剩余的位置。

MindSpore：函数入参`value`支持标量，元素个数为1的numpy数组。

## 代码示例

以输入为标量为例，代码样例如下：

TensorFlow:

```python
import tensorflow as tf

init = tf.keras.initializers.Constant(2)

x = init(shape=(2, 4))
y = init(shape=(3, 4))

with tf.Session() as sess:
    print(x.eval(), "\n")
    print(y.eval())

# out:
# [[2. 2. 2. 2.]
#  [2. 2. 2. 2.]]

# [[2. 2. 2. 2.]
#  [2. 2. 2. 2.]
#  [2. 2. 2. 2.]]
```

MindSpore:

```python
import mindspore as ms
from mindspore.common.initializer import initializer, Constant

x = initializer(Constant(2), shape=[2, 4], dtype=ms.float32)

print(x)

# out:
# [[2. 2. 2. 2.]
#  [2. 2. 2. 2.]]
```
