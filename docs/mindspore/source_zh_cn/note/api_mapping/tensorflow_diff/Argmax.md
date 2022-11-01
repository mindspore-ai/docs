# 比较与tf.math.argmax的功能差异

## tf.math.argmax

```text
tf.math.argmax(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None,
) -> Tensor
```

更多内容详见 [tf.math.argmax](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/argmax?hl=zh-cn)。

## mindspore.ops.Argmax

```text
class mindspore.ops.Argmax(axis=-1, output_type=mstype.int32)(input_x) -> Tensor
```

更多内容详见 [mindspore.ops.Argmax](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Argmin.html)。

## 差异对比

TensorFlow：返回Tensor沿着给定的维度上最大值的索引，返回值类型默认为tf.int64，默认是返回axis为0时最大值的索引。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，返回值类型默认为ms.int32，默认是返回axis为-1时最大值的索引。

| 分类 | 子类 | PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|输入 | 单输入 | input | input_x |都是输入Tensor，二者均不支持0维张量，TensorFlow支持Tensor类型和Numpy.ndarray类型的输入，MindSpore只支持Tensor类型的输入 |
|参数 | 参数1 | axis | axis |功能一致，参数名相同，默认值不同 |
| | 参数2 | output_type | output_type | 功能一致，参数名相同，默认值不同 |
| | 参数3 | name | - | 功能一致，MindSpore无此参数，行为与TensorFlow算子参数name设为None时一致 |

### 代码示例1

> TensorFlow的argmax算子在不显式给出axis参数时，计算结果是axis按默认值为0时最大值的索引，而MindSpore默认是返回axis为-1时最大值的索引。因此，为了得到相同的计算结果，在计算前，将mindspore.ops.Argmax算子参数axis赋值为0，同时为保证二者输出类型是一致的，需使用[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Cast.html)算子将MindSpore的计算结果转换成mindspore.int64。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
tf_output = tf.math.argmax(tf.constant(x))
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 0
ms_argmax = mindspore.ops.Argmax(axis)
ms_output = ms_argmax(Tensor(x))
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
```

### 代码示例2

> TensorFlow和MindSpore参数传入方式不会影响功能。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
axis = 2
tf_output = tf.math.argmax(tf.constant(x), axis)
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[3 3 3]
#  [3 3 3]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 2
ms_argmax = mindspore.ops.Argmax(axis)
ms_output = ms_argmax(Tensor(x))
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[3 3 3]
#  [3 3 3]]
```

### 代码示例3

> TensorFlow参数output_type用于指定输出数据类型，默认是tf.int64。而MindSpore的参数output_type默认值是mindspore.int32，为保证二者输出类型是一致的，需使用[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Cast.html)算子将MindSpore的计算结果转换成mindspore.int64。TensorFlow参数name用于定义执行操作的名称，不影响结果，MindSpore无此参数。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
axis = 1
tf_output = tf.math.argmax(tf.constant(x), axis, name="tf_output")
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[2 2 2 2]
#  [2 2 2 2]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 1
ms_argmax = mindspore.ops.Argmax(axis)
ms_output = ms_argmax(Tensor(x))
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[2 2 2 2]
#  [2 2 2 2]]
```
