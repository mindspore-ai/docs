# Function Differences with tf.nn.softmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Softmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.softmax

```text
tf.nn.softmax(logits, axis=None, name=None) -> Tensor
```

For more information, see [tf.nn.softmax](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax).

## mindspore.nn.Softmax

```text
class mindspore.nn.Softmax(axis=-1)(x) -> Tensor
```

For more information, see [mindspore.nn.Softmax](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Softmax.html).

## Differences

TensorFlow: a generalization of the binary classification function on multiclassification, which aims to present the results of multiclassification in the form of probabilities.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | parameter 1 | logits     | x      | Same function, different parameter names |
|      | parameter 2 | axis       | axis      | -        |
|      | parameter 3 | name       | -      | Not involved       |

### Code Example

> The two APIs achieve the same function and have the same usage.

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
