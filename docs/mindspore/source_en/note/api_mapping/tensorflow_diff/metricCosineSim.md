# Function Differences with tf.keras.metrics.CosineSimilarity

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/metricCosineSim.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.metrics.CosineSimilarity

```python
tf.keras.metrics.CosineSimilarity(
    name='cosine_similarity', dtype=None, axis=-1
)
```

For more information, see [tf.keras.metrics.CosineSimilarity](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/CosineSimilarity).

## mindspore.train.CosineSimilarity

```python
mindspore.train.CosineSimilarity(similarity="cosine", reduction="none", zero_diagonal=True)
```

For more information, see [mindspore.train.CosineSimilarity](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.CosineSimilarity.html#mindspore.train.CosineSimilarity).

## Usage

MindSpore: The input is a matrix, each row of the matrix can be regarded as a sample, and the return value is the similarity matrix. If `similarity="cosine"`, it is cosine similarity calculation logic, same as `tf.keras.metrics.CosineSimilarity` calculation logic, and if `similarity="dot"`, it is matrix dot product transpose matrix. `reduction` can be set to `none`, `sum`, `mean`, which correspond to the original result matrix, sum and average calculation respectively.

TensorFlow: The inputs are the predicted and true values, which are computed by cosine similarity = (a . b) / ||a|| ||b|| is computed and the return result is the mean value of cosine similarity for all data streams.

## Code Example

```python
import tensorflow as tf
tf.enable_eager_execution()

m = tf.keras.metrics.CosineSimilarity(axis=1)
m.update_state([[1, 3, 4]], [[2, 4, 2]])
print(m.result().numpy())

# output: 0.8807048


from mindspore.train import CosineSimilarity
import numpy as np

input_data = np.array([[1, 3, 4], [2, 4, 2], [0, 1, 0]])
metric = CosineSimilarity()
metric.update(input_data)
print(metric.eval())

# output:
# [[0.         0.88070485 0.58834841]
#  [0.88070485 0.         0.81649658]
#  [0.58834841 0.81649658 0.        ]]
```
