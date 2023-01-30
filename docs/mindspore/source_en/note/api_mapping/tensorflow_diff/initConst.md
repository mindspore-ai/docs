# Function Differences with tf.keras.initializers.Constant

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/initConst.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.initializers.Constant

```python
tf.keras.initializers.Constant(value=0)
```

For more information, see [tf.keras.initializers.Constant](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/initializers/Constant).

## mindspore.common.initializer.Constant

```python
mindspore.common.initializer.Constant(value)
```

For more information, see [mindspore.common.initializer.Constant](https://mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant).

## Usage

TensorFlow: The function input parameter `value` supports scalar, list, tuple, and array types. Suppose you need to create a tensor of the specified shape, and the input parameter `value` of this interface is a list or an array, the number of elements contained in `value` must be less than or equal to the number of elements with the specified shape. If the number of elements contained in `value` must be less than the number of elements with the specified shape, the last element of `value` is used to fill the remaining positions.

MindSpore: The function input parameter `value` supports scalar and array types. When `value` is an array, only a tensor with the same shape as `value` can be generated.

## Code Example

As an example, if the input is an array, the code sample is as follows:

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


