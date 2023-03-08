# 比较与tf.raw_ops.LRN的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/LRN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.raw_ops.LRN

```text
tf.raw_ops.LRN(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None) -> Tensor
```

更多内容详见[tf.raw_ops.LRN](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/LRN)。

## mindspore.ops.LRN

```text
mindspore.ops.LRN(x, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS") -> Tensor
```

更多内容详见[mindspore.ops.LRN](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.LRN.html)。

## 差异对比

TensorFlow：进行局部响应归一化操作，返回一个与input具有相同类型的Tensor。

MindSpore：MindSpore此API实现功能与TensorFlow一致，部分参数名不同，且多一个参数指定归一化区域norm_region。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x        | 功能一致，参数名不同           |
|  | 参数2 | depth_radius       | depth_radius         | - |
|  | 参数3 | bias       | bias         | - |
|  | 参数4 | alpha       | alpha         | - |
|  | 参数5 | beta       | beta         | - |
|  | 参数6 | -       | norm_region         | 指定归一化区域。TensorFlow无此参数 |
| | 参数7 | name | -           | 不涉及 |

### 代码示例1

MindSpore和TensorFlow输出结果一致。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

input_x = tf.constant(np.array([[[[0.1], [0.2]],[[0.3], [0.4]]]]), dtype=tf.float32)

output = tf.raw_ops.LRN(input=input_x, depth_radius=1, bias=0.00001, alpha=0.0000001, beta=0.00001)
print(output.numpy())
# [[[[0.10001152]
#     [0.2002304]]
#     [[0.3003455]
#     [0.40004608]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as ops
import numpy as np

input_x = Tensor(np.array([[[[0.1], [0.2]],[[0.3], [0.4]]]]), mindspore.float32)
lrn = ops.LRN(depth_radius=1, bias=0.00001, alpha=0.0000001, beta=0.00001)
output = lrn(input_x)
print(output)
# [[[[0.10001152]
#     [0.2002304]]
#     [[0.3003455]
#     [0.40004608]]]]
```
