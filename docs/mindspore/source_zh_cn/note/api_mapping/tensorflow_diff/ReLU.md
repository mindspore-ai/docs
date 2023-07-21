# 比较与tf.nn.relu的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/ReLU.md)

## tf.nn.relu

```text
tf.nn.relu(features, name=None) -> Tensor
```

更多内容详见[tf.nn.relu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/relu)。

## mindspore.nn.ReLU

```text
class mindspore.nn.ReLU()(x) -> Tensor
```

更多内容详见[mindspore.nn.ReLU](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.ReLU.html)。

## 差异对比

TensorFlow：ReLU激活函数。

MindSpore：MindSpore此算子实现功能与TensorFlow一致，参数名不同，且算子需要先实例化。

| 分类 | 子类 | TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | features | x | 功能一致，参数名不同 |
| | 参数2 | name | - | 不涉及 |

### 代码示例

> 两API实现功能一致，但TensorFlow该算子是函数式的，可以直接接受输入。MindSpore中需要先实例化。

```python
# TensorFlow
import tensorflow as tf

x = tf.constant([[-1.0, 2.2], [3.3, -4.0]], dtype=tf.float16)
out = tf.nn.relu(x).numpy()
print(out)
# [[0.  2.2]
#  [3.3 0. ]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float16)
relu = nn.ReLU()
output = relu(x)
print(output)
# [[0.  2.2]
#  [3.3 0. ]]
```