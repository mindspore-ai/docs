# 比较与tf.math.argmin的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/argmin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.math.argmin

```text
tf.math.argmin(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None,
) -> Tensor
```

更多内容详见[tf.math.argmin](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/argmin?hl=zh-cn)。

## mindspore.ops.argmin

```text
mindspore.ops.argmin(x, axis=None, keepdims=False) -> Tensor
```

更多内容详见[mindspore.ops.argmin](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.argmin.html)。

## 差异对比

TensorFlow：返回Tensor沿着给定的维度上最小值的索引，返回值类型默认为tf.int64，默认是返回axis为0时最小值的索引。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，返回值类型默认为ms.int32，默认是返回axis为-1时最小值的索引。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|输入 | 单输入 | input | x |都是输入Tensor，二者均不支持零维张量，TensorFlow支持Tensor类型和Numpy.ndarray类型的输入，MindSpore只支持Tensor类型的输入 |
|参数 | 参数1 | axis | axis |功能一致，参数名不同，默认值不同 |
|     | 参数2 | output_type | -| 指定输出类型，MindSpore无此参数 |
|     | 参数3 | name | - | 不涉及 |
| | 参数4 | - | keepdims | TensorFlow无此参数，MindSpore的参数keepdims为True时将进行聚合的维度保留，并设定为1 |

### 代码示例1

> TensorFlow的argmin算子在不显式给出axis参数时，计算结果是axis按默认值为0时最小值的索引，而MindSpore默认是返回axis为-1时最小值的索引。因此，为了得到相同的计算结果，在计算前，将mindspore.ops.argmin算子参数axis赋值为0，同时为保证二者输出类型是一致的，需使用[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Cast.html)算子将MindSpore的计算结果转换成mindspore.int64。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmin = tf.math.argmin
tf_output = tf.math.argmin(tf.constant(x))
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 0
ms_argmin = mindspore.ops.argmin
ms_output = ms_argmin(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]
```

### 代码示例2

> TensorFlow和MindSpore参数传入方式不会影响功能。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmin = tf.math.argmin
axis = 2
tf_output = tf.math.argmin(tf.constant(x), axis)
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[0 0 0]
#  [0 0 0]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 2
ms_argmin = mindspore.ops.argmin
ms_output = ms_argmin(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[0 0 0]
#  [0 0 0]]
```

### 代码示例3

> TensorFlow参数output_type用于指定输出数据类型，默认是tf.int64。而MindSpore的参数output_type默认值是mindspore.int32，为保证二者输出类型是一致的，需使用[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Cast.html)算子将MindSpore的计算结果转换成mindspore.int64。TensorFlow参数name用于定义执行操作的名称，不影响结果，MindSpore无此参数。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmin = tf.math.argmin
axis = 1
tf_output = tf.math.argmin(tf.constant(x), axis, name="tf_output")
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[0 0 0 0]
#  [0 0 0 0]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 1
ms_argmin = mindspore.ops.argmin
ms_output = ms_argmin(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[0 0 0 0]
#  [0 0 0 0]]
```
