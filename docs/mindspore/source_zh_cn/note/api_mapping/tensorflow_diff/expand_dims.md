# 比较与tf.expand_dims的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/expand_dims.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.expand_dims

```text
tf.expand_dims(x, axis, name=None) -> Tensor
```

更多内容详见 [tf.expand_dims](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/expand_dims)。

## mindspore.ops.expand_dims

```text
mindspore.ops.expand_dims(input_x, axis) -> Tensor
```

更多内容详见 [mindspore.ops.expand_dims](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.expand_dims.html)。

## 差异对比

TensorFlow：对输入x在给定的轴上添加额外维度。

MindSpore：MindSpore此API实现功能与TensorFlow一致，仅参数名不同。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                  |
| ---- | ----- | ---------- | --------- | --------------------- |
| 参数 | 参数1 | x          | input_x   | 功能一致，参数名不同 |
|      | 参数2 | axis       | axis      | - |
|      | 参数3 | name       | -      | 不涉及 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
axis = 1
out = tf.expand_dims (x, axis).numpy()
print(out)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]

# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
axis = 1
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]

```

### 代码示例2

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = np.array([[1,1,1]], dtype=np.float32)
axis = 2
out = tf.expand_dims (x, axis).numpy()
print(out)
# [[[1.]
#   [1.]
#   [1.]]]


# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1,1,1]]), mindspore.float32)
axis = 2
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[1.]
#   [1.]
#   [1.]]]
```
