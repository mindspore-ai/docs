# Function Differences with tf.nn.dropout

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.dropout

```python
tf.nn.dropout(
    x,
    rate,
    noise_shape=None,
    seed=None,
    name=None
) -> Tensor
```

For more information, see [tf.nn.dropout](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/dropout).

## mindspore.ops.dropout

```python
mindspore.ops.dropout(x, p=0.5, seed0=0, seed1=0) -> Tensor
```

For more information, see [mindspore.ops.dropout](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.dropout.html).

## Differences

TensorFlow: dropout is a function used to prevent or mitigate overfitting by dropping a random portion of neurons at different training sessions. That is, the neuron output is randomly set to 0 with a certain probability p, which serves to reduce the neuron correlation. The remaining parameters that are not set to 0 will be scaled with $\frac{1}{1-rate}$.

MindSpore: MindSpore API basically implements the same function as TensorFlow, but TensorFlow has an additional noise_shape parameter to control the retention/discard dimension.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| ---- | ----- | ----------- | --------- | ------------------------------------------------------------ |
| Parameters  | Parameter 1 | x           | x         | -                                                            |
|      | Parameter 2 | rate        | p         | Same function, different parameter names                                         |
|      | Parameter 3 | noise_shape |     -      | A 1 int32 tensor representing a randomly generated "keep/drop" flag for the shape. MindSpore does not have this parameter  |
|      | Parameter 4 | seed        | seed0     | Same function, different parameter names                                         |
|      | Parameter 5 | name        |           | Not involved                                                      |
|      | Parameter 6 |      -       | seed1     | The global random seed, which together with the random seed of the operator layer determines the final generated random number. Default value: 0 |

### Code Example 1

> When the value of noise_shape is None, the two APIs functions are the same.

```python
# TensorFlow
import tensorflow as tf
import numpy as np
neuros = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],dtype=np.float32)
neuros_drop = tf.nn.dropout(neuros, rate=0.2)
print(neuros_drop.shape)
# (10, 10)

# MindSpore
import mindspore
x = mindspore.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], mindspore.float32)
output, mask = mindspore.ops.dropout(x, p=0.2)
print(output.shape)
# (10, 10)
```


