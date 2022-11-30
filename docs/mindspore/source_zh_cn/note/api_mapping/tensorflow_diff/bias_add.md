# 比较与tf.nn.bias_add的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/bias_add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.nn.bias_add

```text
class tf.nn.bias_add(value, bias, data_format=None, name=None)
```

更多内容详见 [tf.nn.bias_add](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/bias_add?hl=zh-cn%3B)。

## mindspore.ops.bias_add

```text
mindspore.ops.bias_add(input_x, bias)
```

更多内容详见 [mindspore.ops.bias_add](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bias_add.html?highlight=bias_add)。

## 差异对比

TensorFlow：返回输入value和bias的tensor相加之和，其中bias被限制为1D的tensor，value支持各种数量的维度，两者相加前会把bias广播成与输入value的shape一致。

MindSpore: MindSpore此API实现功能与TensorFlow基本一致，不过MindSpore的输入input_x只支持2-5维的shape。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                                  |
| ---- | ----- | ---------- | --------- | ------------------------------------- |
| 参数 | 参数1 | value      | input_x   | 功能一致，参数名不同                  |
|      | 参数2 | bias       | bias      | 功能一致                              |
|      | 参数3 | data_format | -         | 输入数据的数据格式，MindSpore无此参数 |
|      | 参数4 | name       | -         | 不涉及   |

### 代码示例1

两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
value = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
bias = tf.constant([-2, -1], dtype=tf.float32)
result = tf.nn.bias_add(value, bias)
ss = tf.compat.v1.Session()
output = ss.run(result)
print(output)
# [[-1.  1.]
#  [ 1.  3.]
#  [ 3.  5.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

input_x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
bias = Tensor(np.array([-2 , -1]), mindspore.float32)
output = ops.bias_add(input_x, bias)
print(output)
# [[-1.  1.]
#  [ 1.  3.]
#  [ 3.  5.]]
```

