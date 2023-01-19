# Function Differences with tf.math.add

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.add

```text
tf.math.add(x, y, name=None) -> Tensor
```

For more information, see [tf.math.add](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/add?hl=zh-cn%3B).

## mindspore.ops.add

```text
mindspore.ops.add(x, y) -> Tensor
```

For more information, see [mindspore.ops.add](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.add.html).

## Differences

TensorFlow: Computes the element sum of input x and input y, and return a Tensor with the same type as x.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | x | x        | -                                 |
|  | Parameter 2 | y       | y         | - |
| | Parameter 3 | name | -           | Not involved |

### Code Example 1

MindSpore and TensorFlow output the same result when both x and y inputs are Tensor and the data types are the same.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[1,2]]).astype(np.float32))
y = tf.constant(np.array([[1],[2]]).astype(np.float32))
output = tf.math.add(x, y)
print(output.numpy())
# [[2. 3.]
#  [3. 4.]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]).astype(np.float32))
y = Tensor(np.array([[1], [2]]).astype(np.float32))
output = mindspore.ops.add(x, y)
print(output.asnumpy())
# [[2. 3.]
#  [3. 4.]]
```

### Code Example 2

TensorFlow supports scalar summation and the x and y data types must be the same. MindSpore version 1.8.1 does not support scalar summation at this time, but the x and y data types can be different. In order to get the same result, the scalar is converted to Tensor for calculation.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.array([[1,2]]).astype(np.float32)
y = np.array([[1],[2]]).astype(np.float32)
output = tf.math.add(x, y)
print(output.numpy())
# [[2. 3.]
#  [3. 4.]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]).astype(np.int32))
y = Tensor(np.array([[1], [2]]).astype(np.float32))
output = mindspore.ops.add(x, y)
print(output.asnumpy())
# [[2. 3.]
#  [3. 4.]]
```

### cOde Example 3

The name parameter of TensorFlow is used to define the name of the operation and has no effect on the calculation result.

```python
# TensorFlow
from unicodedata import name
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[1,2]]).astype(np.float32))
y = tf.constant(np.array([[1],[2]]).astype(np.float32))
output = tf.math.add(x, y, name="add")
print(output.numpy())
# [[2. 3.]
#  [3. 4.]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([1, 2]).astype(np.float32))
y = Tensor(np.array([[1], [2]]).astype(np.float32))
output = mindspore.ops.add(x, y)
print(output.asnumpy())
# [[2. 3.]
#  [3. 4.]]
```
