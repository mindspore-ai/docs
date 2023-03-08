# 比较与tf.compat.v1.assign_sub的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/assign_sub.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.compat.v1.assign_sub

```text
tf.compat.v1.assign_sub(ref, value, use_locking=None, name=None) -> Tensor
```

更多内容详见[tf.compat.v1.assign_sub](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/assign_sub)。

## mindspore.ops.assign_sub

```text
mindspore.ops.assign_sub(variable, value)-> Tensor
```

更多内容详见[mindspore.ops.assign_sub](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.assign_sub.html)。

## 差异对比

TensorFlow：从网络参数减去特定数值来更新网络参数，返回一个与ref具有相同类型的Tensor。

MindSpore：MindSpore此API实现功能与TensorFlow一致，部分参数名不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | ref | variable        | 功能一致，参数名不同           |
|  | 参数2 | value       | value          | - |
|  | 参数3 | use_locking       | -         | TensorFlow中为是否在更新操作中使用锁，默认值：False。MindSpore无此参数 |
|  | 参数4 | name | -           | 不涉及 |

### 代码示例1

MindSpore和TensorFlow输出结果一致。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

variable = tf.Variable(np.array([[2.4, 1], [0.1, 6]]), dtype=tf.float32)
value = tf.constant(np.array([[-2, 3], [3.6, 1]]), dtype=tf.float32)
out = tf.compat.v1.assign_sub(variable, value)
print(out.numpy())
# [[ 4.4 -2. ]
#  [-3.5  5. ]]

# MindSpore
import mindspore
import numpy as np
from mindspore.ops import function as ops
from mindspore import Tensor

variable = Tensor(np.array([[2.4, 1], [0.1, 6]]), mindspore.float32)
value = Tensor(np.array([[-2, 3], [3.6, 1]]), mindspore.float32)
out = ops.assign_sub(variable, value)
print(out)
# [[ 4.4 -2. ]
#  [-3.5  5. ]]
```
