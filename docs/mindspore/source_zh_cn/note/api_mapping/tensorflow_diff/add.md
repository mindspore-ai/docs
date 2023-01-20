# 比较与tf.math.add的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.add

```text
tf.math.add(x, y, name=None) -> Tensor
```

更多内容详见[tf.math.add](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/math/add?hl=zh-cn%3B)。

## mindspore.ops.add

```text
mindspore.ops.add(x, y) -> Tensor
```

更多内容详见[mindspore.ops.add](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.add.html)。

## 差异对比

TensorFlow：计算输入x和输入y的元素和，返回一个与x具有相同类型的Tensor。

MindSpore：MindSpore此API实现功能与TensorFlow一致，仅参数名不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | x | x        | -                                 |
|  | 参数2 | y       | y         | - |
| | 参数3 | name | -           | 不涉及 |

### 代码示例1

当x和y输入都为Tensor且数据类型一致时，MindSpore和TensorFlow输出结果一致。

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

### 代码示例2

TensorFlow支持标量相加，且x和y数据类型必须保持一致，MindSpore 1.8.1版本暂不支持标量相加,，但x和y数据类型可以不同。为了得到相同的结果，将标量转化为Tensor进行计算。

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

### 代码示例3

TensorFlow的name参数用于定义操作的名称，对计算结果不影响。

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
