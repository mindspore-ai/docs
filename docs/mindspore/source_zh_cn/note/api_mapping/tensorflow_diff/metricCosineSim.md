# 比较与tf.keras.metrics.CosineSimilarity的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/metricCosineSim.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.metrics.CosineSimilarity

```python
tf.keras.metrics.CosineSimilarity(
    name='cosine_similarity', dtype=None, axis=-1
)
```

更多内容详见[tf.keras.metrics.CosineSimilarity](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/CosineSimilarity)。

## mindspore.nn.CosineSimilarity

```python
mindspore.nn.CosineSimilarity(similarity="cosine", reduction="none", zero_diagonal=True)
```

更多内容详见[mindspore.nn.CosineSimilarity](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CosineSimilarity.html#mindspore.nn.CosineSimilarity)。

## 使用方式

MindSpore： 输入为矩阵，矩阵每行可看做一条样本，返回值为相似度矩阵。若`similarity="cosine"`，则为cosine相似度计算逻辑，与`tf.keras.metrics.CosineSimilarity`计算逻辑相同，若`similarity="dot"`，则为矩阵点乘转置矩阵。`reduction`可设置"none"、'sum'、 'mean'，分别对应原始结果矩阵，求和和求平均计算。

TensorFlow：输入为预测值和真实值，通过cosine similarity = (a . b) / ||a|| ||b||进行计算，返回结果为所有数据流的cosine相似度均值。

## 代码示例

```python
import tensorflow as tf
tf.enable_eager_execution()

m = tf.keras.metrics.CosineSimilarity(axis=1)
m.update_state([[1, 3, 4]], [[2, 4, 2]])
print(m.result().numpy())

# output: 0.8807048


from mindspore import nn
import numpy as np

input_data = np.array([[1, 3, 4], [2, 4, 2], [0, 1, 0]])
metric = nn.CosineSimilarity()
metric.update(input_data)
print(metric.eval())

# output:
# [[0.         0.88070485 0.58834841]
#  [0.88070485 0.         0.81649658]
#  [0.58834841 0.81649658 0.        ]]
```
