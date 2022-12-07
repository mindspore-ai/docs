# 比较与tf.nn.leaky_relu的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/LeakyReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.nn.leaky_relu

```text
tf.nn.leaky_relu(features, alpha=0.2, name=None) -> Tensor
```

更多内容详见 [tf.nn.leaky_relu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/leaky_relu)。

## mindspore.nn.LeakyReLU

```text
class mindspore.nn.LeakyReLU(alpha=0.2)(x) -> Tensor
```

更多内容详见 [mindspore.nn.LeakyReLU](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LeakyReLU.html)。

## 差异对比

TensorFlow：应用Leaky ReLU激活函数，其中参数`alpha`是用于控制激活函数的斜率。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，仅参数名不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | features | x |功能一致，参数名不同 |
| | 参数2 | alpha | alpha | - |
| | 参数3 | name | - | 不涉及 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

features = tf.constant([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=tf.float32)
output = tf.nn.leaky_relu(features).numpy()
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
output = m(x)
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```
