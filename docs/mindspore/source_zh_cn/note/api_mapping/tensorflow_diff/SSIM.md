# 比较与tf.image.ssim的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SSIM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.image.ssim

```text
tf.image.ssim(
    img1,
    img2,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03
) -> Tensor
```

更多内容详见[tf.image.ssim](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/image/ssim?hl=zh-cn)。

## mindspore.nn.SSIM

```text
class mindspore.nn.SSIM(
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03
)(img1, img2) -> Tensor
```

更多内容详见[mindspore.nn.SSIM](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.SSIM.html)。

## 差异对比

TensorFlow: 在TensorFlow中，算子是函数式的，直接接收输入的两张图片并返回结果。`max_val`参数不可缺省。接受图片的格式为"NHWC"。

MindSpore: 在MindSpore中，算子需要先实例化，然后接收输入返回结果。接受图片的格式为"NCHW"。

| 分类 | 子类  | TensorFlow   | MindSpore    | 差异                                                         |
| ---- | ----- | ------------ | ------------ | ------------------------------------------------------------ |
| 参数 | 参数1 | img1         | img1         | MindSpore在实例化的函数中接收该输入，功能上一致            |
|      | 参数2 | img2         | img2         |MindSpore在实例化的函数中接收该输入，功能上一致            |
|      | 参数3 | max_val      | max_val      | tensorflow中此参数是必须的，mindspore中此参数可缺省，默认值为`1.0` |
|      | 参数4 | filter_size  | filter_size  | -                                                            |
|      | 参数5 | filter_sigma | filter_sigma | -                                                            |
|      | 参数6 | k1           | k1           | -                                                            |
|      | 参数7 | k2           | k2           | -                                                            |

## 差异分析与示例

### 代码示例1

> TensorFlow中参数`max_val`是必须的，MindSpore中可以缺省，默认值为`1.0`.

```python
# tensorflow
import numpy as np
import tensorflow as tf

img1 = tf.ones([1, 16, 16, 3])
img2 = tf.ones([1, 16, 16, 3])
output = tf.image.ssim(img1, img2, max_val=1.0)
print(output.numpy())
# [1.]

# MindSpore
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

net = nn.SSIM()
img1 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
img2 = Tensor(np.ones([1, 3, 16, 16]).astype(np.float32))
output = net(img1, img2)
print(output)
# [1.]
```

### 代码示例2

> TensorFlow中接受图片输入的格式为“NHWC”，MindSpore中为"NCHW"。下面的例子用同一个随机种子生成的4D张量，在MindSpore中经`(0, 3, 1, 2)`的轴变换后可以得到和TensorFlow相同的结果。

```python
# TensorFlow
import numpy as np
import tensorflow as tf

np.random.seed(10)
img1 = np.random.randint(0, 2, (2, 5, 5, 5)).astype(np.float32)
img2 = np.random.randint(0, 2, (2, 5, 5, 5)).astype(np.float32)
output = tf.image.ssim(img1, img2, max_val=1, filter_size=3)
print(output.numpy())
# [-0.00746754 -0.09539266]

# MindSpore
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor

np.random.seed(10)
img1 = np.random.randint(0, 2, (2, 5, 5, 5)).astype(np.float32)
img2 = np.random.randint(0, 2, (2, 5, 5, 5)).astype(np.float32)
img1_t = Tensor(img1)
img2_t = Tensor(img2)
net = nn.SSIM(filter_size=3)
transpose = ops.Transpose()
trans_term = (0, 3, 1, 2)
img1_trans = transpose(img1_t, trans_term)
img2_trans = transpose(img2_t, trans_term)
output_m = net(img1_trans, img2_trans)
print(output_m)
# [-0.00746753 -0.09539266]
```
