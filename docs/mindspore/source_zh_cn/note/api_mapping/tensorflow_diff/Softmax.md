# 比较与tf.nn.softmax的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.nn.softmax

```text
tf.nn.softmax(logits, axis=None, name=None) -> Tensor
```

更多内容详见[tf.nn.softmax](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax)。

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1)(x) -> Tensor
```

更多内容详见[mindspore.nn.Softmax](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Softmax.html)。

## 差异对比

TensorFlow：它是二分类函数在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。

MindSpore：MindSpore此API实现功能与TensorFlow一致，仅参数名不同。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                                                     |
| ---- | ----- | ---------- | --------- | -------------------------------------------------------- |
| 参数 | 参数1 | logits     | x        | 功能一致，参数名不同 |
|      | 参数2 | axis       | axis      | -                                   |
|      | 参数3 | name       | -      | 不涉及                               |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = tf.constant([-1, -2, 0, 2, 1], dtype=tf.float16)
output = tf.nn.softmax(x)
print(output.numpy())
# [0.03168 0.01165 0.0861  0.636   0.2341 ]

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
softmax = mindspore.nn.Softmax()
output = softmax(x)
print(output)
# [0.03168 0.01165 0.0861  0.636   0.2341 ]
```
