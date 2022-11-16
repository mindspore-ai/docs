# 比较与tf.nn.dropout的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见 [tf.nn.dropout](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/dropout)

## mindspore.ops.dropout

```python
mindspore.ops.dropout(x, p=0.5, seed0=0, seed1=0) -> Tensor
```

更多内容详见 [mindspore.ops.dropout](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dropout.html)

## 差异对比

TensorFlow：dropout是为了防止或减轻过拟合而使用的函数，它会在不同的训练过程中随机丢弃一部分神经元。也就是以一定的概率p随机将神经元输出设置为0，起到减小神经元相关性的作用。其余未被设置为0的参数将会以$\frac{1}{1-p}$进行缩放。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，不过TensorFlow多了一个控制保留/丢弃维度的noise_shape参数。

| 分类 | 子类  | TensorFlow  | MindSpore | 差异                                                         |
| ---- | ----- | ----------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | x           | x         | -                                                            |
|      | 参数2 | rate        | p         | 功能一致，参数名不同                                         |
|      | 参数3 | noise_shape |           | 一个1为的int32张量，代表了随机产生“保留/丢弃“标志的shape。mindspore无此参数 |
|      | 参数4 | seed        | seed0     | 功能一致，参数名不同                                         |
|      | 参数5 | name        |           | 此操作的名称（可选）。功能一致，MindsSpore无此参数           |
|      | 参数6 |             | seed1     | 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值：0 |

### 代码示例1

> 有时候我们希望对dropout的方式做一些约束，为了实现这些需求，需要用到dropout中的nose_shape参数。noise_shape是一个一维张量，长度必须跟x.shape一致。而且noise_shape里边的元素，只能是1或者是x.shape里对应的元素。当哪个轴为1时，这个轴就会被一致地dropout。TensorFlow有这项功能，而MindSpore里没有这项功能。

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
neuros_drop = tf.nn.dropout(neuros, rate=0.2, noise_shape=(1, 10))
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
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], mindspore.float64)
output, mask = mindspore.ops.dropout(x, p=0.2)
print(output.shape)
# (10, 10)
```

### 示例代码2

> 当noise_shape的值为None时，两API功能一致

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
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], mindspore.float64)
output, mask = mindspore.ops.dropout(x, p=0.2)
print(output.shape)
# (10, 10)
```
