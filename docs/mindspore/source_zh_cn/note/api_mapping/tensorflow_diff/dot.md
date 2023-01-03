# 比较与tf.keras.backend.dot的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.backend.dot

```text
tf.keras.backend.dot(x, y) -> Tensor
```

更多内容详见[tf.keras.backend.dot](https://keras.io/zh/backend/#dot)。

## mindspore.ops.dot

```text
mindspore.ops.dot(x1, x2) -> Tensor
```

更多内容详见[mindspore.ops.dot](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.dot.html)。

## 差异对比

TensorFlow：计算两个Tensor或Variable之间的点积。

MindSpore：当输入的两个参数都是张量时，MindSpore此API实现功能与TensorFlow一致，仅参数名不同。当输入的两个参数有任何一个是变量时，仅TensorFlow支持。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | x | x1 |功能一致，参数名不同，且MindSpore参数只能为Tensor类型 |
| | 参数2 | y | x2 |功能一致，参数名不同，且MindSpore参数只能为Tensor类型 |

### 代码示例

> 当两个输入参数的都为Tensor类型时，实现功能一致，用法相同。

```python
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.ones([1, 3, 2])
xy = tf.keras.backend.dot(x, y)
print(xy.numpy())
# [[[3. 3.]]
#  [[3. 3.]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x1 = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
x2 = Tensor(np.ones(shape=[1, 3, 2]), mindspore.float32)
out = mindspore.ops.dot(x1, x2)
print(out)
# [[[3. 3.]]
#  [[3. 3.]]]
```
