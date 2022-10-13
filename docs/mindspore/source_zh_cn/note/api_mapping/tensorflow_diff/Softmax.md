# 比较与tf.nn.softmax的功能差异

## tf.nn.softmax

```text
tf.nn.softmax(logits, axis=None) -> Tensor
```

更多内容详见 [tf.nn.softmax](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax)。

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1) -> Tensor
```

更多内容详见 [mindspore.nn.Softmax](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Softmax.html)。

## 差异对比

TensorFlow:它是二分类函数，在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。

MindSpore:MindSpore此API实现功能与TensorFlow一致， 仅参数名不同。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                                                     |
| ---- | ----- | ---------- | --------- | -------------------------------------------------------- |
| 参数 | 参数1 | logits     | -         | 非空tensor。必须是以下类型之一：half、float32、float64。 |
|      | 参数2 | axis       | axis      | -                                    |

### 代码示例1

> 两API实现功能一致， 用法相同。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = tf.constant([-1, -2, 0, 2, 1],dtype = tf.float16)
output = tf.nn.softmax(x)
print(output.numpy())
#[0.03168 0.01165 0.0861  0.636   0.2341 ]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
#[0.03168 0.01165 0.0861  0.636   0.2341 ]
```

### 代码示例2

> 两API实现功能一致， 用法相同。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = tf.constant([-1, 0., 1.],dtype = tf.float16)
output = tf.nn.softmax(x)
print(output.numpy())
#[0.09   0.2446 0.665 ]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([-1, 0., 1.]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
#[0.0901 0.2448 0.665 ]
```

