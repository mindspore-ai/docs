# Function Differences with tf.clip_by_value

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.clip_by_value

```text
tf.clip_by_value(t, clip_value_min, clip_value_max, name=None) -> Tensor
```

For more information, see [tf.clip_by_value](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/clip_by_value).

## mindspore.ops.clip_by_value

```text
mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None) -> Tensor
```

For more information, see [mindspore.ops.clip_by_value](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.clip_by_value.html).

## Differences

TensorFlow: Given a tensor t, this operation returns a tensor of the same type and shape as t. Any value less than clip_value_min in t is set to clip_value_min, and any value greater than clip_value_max is set to clip_value_max. When clip_value_min is greater than clip_value_max, the value of the tensor will be set to **clip_value_min**.

MindSpore: When clip_value_min is less than or equal to clip_value_max, MindSpore API implements the same function as TensorFlow. When clip_value_min is greater than clip_value_max, the value of the tensor element will be set to **clip_value_max**.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | t | x | Same function, different parameter names |
| | Parameter 2 | clip_value_min | clip_value_min | - |
| | Parameter 3 | clip_value_max | clip_value_max | - |
| | Parameter 4 | name | - | Not involved |

### Code Example 1

> When clip_value_min is less than or equal to clip_value_max, the two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf
t = tf.constant([[1., 25., 5., 7.], [4., 11., 6., 21.]])
t2 = tf.clip_by_value(t, clip_value_min=5, clip_value_max=22)
print(t2.numpy())
#[[ 5. 22.  5.  7.]
# [ 5. 11.  6. 21.]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clip_by_value(input, clip_value_min=5, clip_value_max=22)
print(output)
#[[ 5. 22.  5.  7.]
# [ 5. 11.  6. 21.]]
```

### Code Example 2

> When clip_value_min is greater than clip_value_max, TensorFlow will set the value of the tensor to **clip_value_min** and MindSpore will set it to **clip_value_max**.

```python
# TensorFlow
import tensorflow as tf
t = tf.constant([[1., 25., 5., 7.], [4., 11., 6., 21.]])
t2 = tf.clip_by_value(t, clip_value_min=22, clip_value_max=5)
print(t2.numpy())
#[[ 22. 22. 22. 22.]
# [ 22. 22. 22. 22.]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clip_by_value(input, clip_value_min=22, clip_value_max=5)
print(output)
#[[ 5. 5. 5. 5.]
# [ 5. 5. 5. 5.]]
```

