# Function Differences with tf.fill

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/fill.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.fill

```text
tf.fill(dims, value, name=None) -> Tensor
```

For more information, see [tf.fill](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/fill).

## mindspore.ops.fill

```text
mindspore.ops.fill(type, shape, value) -> Tensor
```

For more information, see [mindspore.ops.fill](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.fill.html).

## Differences

TensorFlow: ‎ is used to generate a tensor with scalar values.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | dims | shape |Same function, different parameter names |
|  | Parameter 2 | value | value | - |
|  | Parameter 3 | name | - | Not involved |
|  | Parameter 4 | - | type | Specify the data type of the output Tensor |

### Code Example 1

> Both APIs implement the same function. MindSpore only has one more parameter specifying the type of output, and the rest of the parameters are used in the same way.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

dims = np.array([2,3])
value = 9
output = tf.fill(dims, value)
output_m = output.numpy()
print(output_m)
#[[9 9 9]
# [9 9 9]]

# MindSpore
import mindspore
import mindspore.ops as ops

type = mindspore.int32
shape = tuple((2,3))
value = 9
output = ops.fill(type, shape, value)
print(output)
#[[9 9 9]
# [9 9 9]]
```


