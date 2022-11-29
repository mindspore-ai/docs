# 比较与tf.math.erf的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/erf.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.erf

``` text
tf.math.erf(x, name=None) -> Tensor
```

更多内容详见 [tf.math.erf](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/erf)。

## mindspore.ops.erf

``` text
mindspore.ops.erf(x) -> Tensor
```

更多内容详见 [mindspore.ops.erf](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.erf.html)。

## 差异对比

TensorFlow：逐元素计算 x 的高斯误差函数，即 $ \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t $ 。

MindSpore：与TensorFlow实现的功能基本一致，但支持的维度大小有差异。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | x | x |功能一致，支持的维度大小有差异 |
|参数 | 参数2 | name | - |不涉及 |

### 代码示例1

> TensorFlow没有限制x的维度，而MindSpore中x支持的维度必须小于8。当x的维度小于8时，两API功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
out = tf.math.erf(x).numpy()
print(out)
# [[[[[[[0.8427007]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
out = ops.erf(x)
print(out)
# [[[[[[[0.8427007]]]]]]]
```

### 代码示例2

> 当x的维度超过或等于8时，可以通过API组和实现同样的功能。使用ops.reshape将x的维度降为1，然后调用ops.erf进行计算，最后再次使用ops.reshape对得到的结果按照x的原始维度进行升维操作。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
out = tf.math.erf(x).numpy()
print(out)
# [[[[[[[[0.8427007]]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
x_reshaped = ops.reshape(x, (-1,))
out_temp = ops.erf(x_reshaped)
out = ops.reshape(out_temp, x.shape)
print(out)
# [[[[[[[[0.8427007]]]]]]]]
```
