# Function Differences with tf.keras.metrics.Precision

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/metricPrecision.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.metrics.Precision

```python
tf.keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
```

For more information, see [tf.keras.metrics.Precision](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/Precision).

## mindspore.train.Precision

```python
mindspore.train.Precision(eval_type="classification")
```

For more information, see [mindspore.train.Precision](https://mindspore.cn/docs/en/master/api_python/train/mindspore.train.Precision.html#mindspore.train.Precision).

## Usage

The calculation formulas are the same, both are PRECISION = true_positives / (true_positives + false_positives), but the logic for processing the input is different, assuming that the input true value is y_true and the predicted value is y_pred, the differences are as follows:

TensorFlow: TensorFlow version 1.15 of this interface only supports single-label scenarios for binary classification, and eventually returns the mean value of precision. y_true is mapped to a Boolean variable (all but 0 are mapped to 1); y_pred>`thresholds` is considered to predict a positive class, and y_pred<=`thresholds` is considered to predict a negative class. The parameters are roughly as follows:

- `thresholds`: Set the threshold to discriminate the predicted value as correct. The default value is None, and 0.5 will be used to determine whether the predicted value is correct or not.

- `top_k`: The default value is None, and the full amount of samples is used. After the setting of thresholds will be invalid, the sample with the predicted value of topk is used for calculation.

- `class_id`: Specify the category id for single category calculation.

MindSpore: Support single-label and multi-label scenarios. The return result is controlled by the boolean parameter `average`, the default is False, which returns the precision statistics of each category. If set to True, it returns the average value.

Taking the following code as an example, the inputs of the two interfaces are the same case. With default parameters, `mindspore.nn.Precision` parses the input as a 3 classification problem with 5 samples, and outputs a list of length 3 representing the precision of each category, and `tf.keras.metrics.Precision` parses the input as a 2 classification problem with 15 samples and output the mean value of precision for all samples.

## Code Example

```python
import tensorflow as tf
tf.enable_eager_execution()

y_pred = [[0.2, 0.5, 0.1], [0.3, 0.1, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.97], [0.2, 0.6, 0.8]]
y_true = [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]

m = tf.keras.metrics.Precision()
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 0.42857143

m = tf.keras.metrics.Precision(thresholds=0.6)
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 0.5

m = tf.keras.metrics.Precision(top_k=3)
m.update_state(y_true, y_pred)
print(m.result().numpy())
# output: 0.33333334


import numpy as np
from mindspore.train import Precision

x = ms.Tensor(np.array([[0.2, 0.5, 0.1], [0.3, 0.1, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.97], [0.2, 0.6, 0.8]]))
y = ms.Tensor(np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))
metric = Precision('classification')
metric.update(x, y)
print(metric.eval(average=True), metric.eval())

# output: 0.8333333333333334 [0.5 1.  1. ]
```
