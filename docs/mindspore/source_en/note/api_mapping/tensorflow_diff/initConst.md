# Function Differences with tf.keras.initializers.Constant

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initConst.md)

## tf.keras.initializers.Constant

```python
tf.keras.initializers.Constant(value=0)
```

For more information, see [tf.keras.initializers.Constant](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/initializers/Constant).

## mindspore.common.initializer.Constant

```python
mindspore.common.initializer.Constant(value)
```

For more information, see [mindspore.common.initializer.Constant](https://mindspore.cn/docs/en/r2.1/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant).

## Usage

TensorFlow: The function input parameter `value` supports scalar, list, tuple, and array types. Suppose you need to create a tensor of the specified shape, and the input parameter `value` of this interface is a list or an array, the number of elements contained in `value` must be less than or equal to the number of elements with the specified shape. If the number of elements contained in `value` must be less than the number of elements with the specified shape, the last element of `value` is used to fill the remaining positions.

MindSpore: The function input parameter `value` supports scalar and array of one element. When `value` is an array, only a tensor with the same shape as `value` can be generated.

## Code Example

As an example, if the input is a scalar, the code sample is as follows:

TensorFlow:

```python
import numpy as np
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


