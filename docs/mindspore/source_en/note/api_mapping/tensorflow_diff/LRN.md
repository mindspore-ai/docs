# Function Differences with tf.raw_ops.LRN

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/LRN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## tf.raw_ops.LRN

```text
tf.raw_ops.LRN(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None) -> Tensor
```

For more information, see [tf.raw_ops.LRN](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/LRN).

## mindspore.ops.LRN

```text
mindspore.ops.LRN(depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS")(x) -> Tensor
```

For more information, see [mindspore.ops.LRN](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.LRN.html).

## Differences

TensorFlow: Performs a local response normalization operation, and returns a Tensor with the same type as the input.

MindSpore: MindSpore API implements the same functions as TensorFlow, with different parameter names and one more parameter specifying the normalized region norm_region.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | x        | Same function, different parameter names           |
|  | Parameter 2 | depth_radius       | depth_radius         | - |
|  | Parameter 3 | bias       | bias         | - |
|  | Parameter 4 | alpha       | alpha         | - |
|  | Parameter 5 | beta       | beta         | - |
|  | Parameter 6 | -       | norm_region         | Specify the normalized region. TensorFlow does not have this parameter |
| | Parameter 7 | name | -           | Not Involved |

### Code Example 1

The outputs of MindSpore and TensorFlow are consistent.

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
