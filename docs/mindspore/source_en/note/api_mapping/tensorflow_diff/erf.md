# Function Differences with tf.math.erf

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/erf.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.erf

```text
tf.math.erf(x, name=None) -> Tensor
```

For more information, see [tf.math.erf](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/erf).

## mindspore.ops.erf

```text
mindspore.ops.erf(x) -> Tensor
```

For more information, see [mindspore.ops.erf](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erf.html).

## Differences

TensorFlow: Compute the Gaussian error function for x element-wise i.e. $ \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t $ .

MindSpore: MindSpore API basically implements the same function as TensorFlow, but there are differences in the size of the supported dimensions.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | x | x |Same function, difference in size of supported dimensions |
|| Parameter 2 | name | - |Not involved |

### Code Example 1

> TensorFlow does not limit the dimension of x, while MindSpore supports dimensions of x that must be less than 8. When the dimension of x is less than 8, the two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
out = tf.math.erf(x).numpy()
print(out)
# [[[[[[[0.8427007]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
out = ops.erf(x)
print(out)
# [[[[[[[0.8427007]]]]]]]
```

### Code Example 2

> When the dimension of x is more than or equal to 8, the same function can be achieved by API group sum. Use ops.reshape to reduce the dimension of x to 1, then call ops.erf to compute it, and finally use ops.reshape again to up-dimension the obtained result according to the original dimension of x.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
out = tf.math.erf(x).numpy()
print(out)
# [[[[[[[[0.8427007]]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
x_reshaped = ops.reshape(x, (-1,))
out_temp = ops.erf(x_reshaped)
out = ops.reshape(out_temp, x.shape)
print(out)
# [[[[[[[[0.8427007]]]]]]]]
```

