# 比较与tf.keras.backend.batch_dot的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/batch_dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.keras.backend.batch_dot

```text
tf.keras.backend.batch_dot(x, y, axes=None)
```

更多内容详见[tf.keras.backend.batch_dot](https://keras.io/zh/backend/#batch_dot)。

## mindspore.ops.batch_dot

```text
mindspore.ops.batch_dot(x1, x2, axes=None)
```

更多内容详见[mindspore.ops.batch_dot](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.batch_dot.html)。

## 差异对比

TensorFlow：当输入x和y是批量数据时，batch_dot返回x和y的点积。

MindSpore：MindSpore此API实现功能与Keras一致，仅参数名不同。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                 |
| ---- | ----- | ---------- | --------- | -------------------- |
| 参数 | 参数1 | x          | x1        | 功能一致，参数名不同 |
|      | 参数2 | y          | x2        | 功能一致，参数名不同 |
|      | 参数3 | axes       | axes      | -             |

### 代码示例1

两API不带axes参数实现功能一致，用法相同。

```python
# TensorFlow
import keras.backend as K
import tensorflow as tf
import numpy as np

x = K.variable(np.random.randint(10,size=(10,12,4,5)), dtype=tf.float32)
y = K.variable(np.random.randint(10,size=(10,12,5,8)), dtype=tf.float32)
output = K.batch_dot(x, y)
print(output.shape)
# (10, 12, 4, 12, 8)

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x1 = Tensor(np.random.randint(10,size=(10,12,4,5)), mindspore.float32)
x2 = Tensor(np.random.randint(10,size=(10,12,5,8)), mindspore.float32)
output = ops.batch_dot(x1, x2)
print(output.shape)
# (10, 12, 4, 12, 8)
```

### 代码示例2

两API带axes参数实现功能一致，用法相同。

```python
# TensorFlow
import keras.backend as K
import tensorflow as tf
import numpy as np

x = K.variable(np.ones(shape=[2, 2]), dtype=tf.float32)
y = K.variable(np.ones(shape=[2, 3, 2]), dtype=tf.float32)
axes = (1, 2)
output = K.batch_dot(x, y, axes)
print(output.shape)
# (2, 3)

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x1 = Tensor(np.ones(shape=[2, 2]), mindspore.float32)
x2 = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
axes = (1, 2)
output = ops.batch_dot(x1, x2, axes)
print(output.shape)
# (2, 3)
```

