# 比较与tf.clip_by_value的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.clip_by_value

```text
tf.clip_by_value(t, clip_value_min, clip_value_max, name=None) -> Tensor
```

更多内容详见[tf.clip_by_value](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/clip_by_value)。

## mindspore.ops.clip_by_value

```text
mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None) -> Tensor
```

更多内容详见 [mindspore.ops.clip_by_value](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clip_by_value.html)。

## 差异对比

TensorFlow：给定一个张量t，此操作返回一个类型和形状与t相同的张量。t中任何小于clip_value_min的都设置为clip_value_min，任何大于的值clip_value_max都设置为clip_value_max。

MindSpore: MindSpore此API实现功能与TensorFlow基本一致,仅部分参数名不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | t | x |功能一致， 参数名不同 |
| | 参数2 | clip_value_min | clip_value_min | 功能一致 |
| | 参数3 | clip_value_max | clip_value_max |功能一致 |

### 代码示例1

> 两API实功能一致， 用法相同。

```python
# TensorFlow
import tensorflow as tf
t = tf.constant([[1., 25., 5., 7.], [4., 11., 6., 21.]])
t2 = tf.clip_by_value(t, clip_value_min=5, clip_value_max=22)
print(t2.numpy())
#[[ 5. 22.  5.  7.]
# [ 5. 11.  6. 21.]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clip_by_value(input, clip_value_min=5, clip_value_max=22)
print(output)
#[[ 5. 22.  5.  7.]
# [ 5. 11.  6. 21.]]
```
