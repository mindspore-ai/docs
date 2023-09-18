# Function Differences with tf.keras.metrics.Recall

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/metricRecall.md)

## tf.keras.metrics.Recall

```python
tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
```

For more information, see [tf.keras.metrics.Recall](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/Recall).

## mindspore.train.Recall

```python
mindspore.train.Recall(eval_type="classification")
```

For more information, see [mindspore.train.Recall](https://mindspore.cn/docs/en/r2.1/api_python/train/mindspore.train.Recall.html#mindspore.train.Recall).

## Usage

The calculation formulas are the same, both are Recall = true_positives / (true_positives + false_negatives), but the logic for processing the input is different, assuming that the input true value is y_true and the predicted value is y_pred, the differences are as follows:

TensorFlow: TensorFlow version 1.15 of this interface only supports single-label scenarios for binary categories, and eventually returns the mean value of recall. y_true is mapped to a Boolean variable (all but 0 are mapped to 1); y_pred>`thresholds` is considered to predict a positive class, and y_pred<=`thresholds` is considered to predict a negative class. The parameters are roughly as follows:

- `thresholds`: Set the threshold to discriminate the predicted value as correct. The default value is None, and 0.5 will be used to determine whether the predicted value is correct.

- `top_k`: The default value is None, and the full amount of samples is used. After the setting, thresholds will be invalid, the sample with the predicted value of topk is used for calculation.

- `class_id`: Specify the category id for single category calculation.

MindSpore: Support single-label and multi-label scenario, control the return result by boolean parameter `average`. Default is False, and return each category recall statistics value. If set to True, return the average value.

Taking the following code as an example, the inputs of the two interfaces are the same cases, and with default parameters, `mindspore.nn.Recall` parses the input as a 3-sample classification problem with 5 samples. Output a list of length 3 indicating the various types of recalls. `tf.keras.metrics.Recall` parses the input as a 2 classification problem with 15 samples and outputs the recall mean of all samples.

## Code Example

```python
import tensorflow as tf
tf.enable_eager_execution()

y_pred = [[0.2, 0.5, 0.1], [0.3, 0.1, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.97], [0.2, 0.6, 0.8]]
y_true = [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]

m = tf.keras.metrics.Recall()
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 0.6

m = tf.keras.metrics.Recall(thresholds=0.6)
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 0.4

m = tf.keras.metrics.Recall(top_k=2)
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 1.0


import numpy as np
from mindspore.train import Recall

x = ms.Tensor(np.array([[0.2, 0.5, 0.1], [0.3, 0.1, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.97], [0.2, 0.6, 0.8]]))
y = ms.Tensor(np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))
metric = Recall('classification')
metric.update(x, y)
print(metric.eval(average=True), metric.eval())

# output: 0.8333333333333334 [1. 0.5 1. ]
```
