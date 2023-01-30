# Function Differences with tf.keras.metrics.Accuracy, tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, and tf.keras.metrics.SparseCategoricalAccuracy

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/metricAcc.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.metrics.Accuracy

```python
tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)
```

For more information, see [tf.keras.metrics.Accuracy](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/Accuracy).

## tf.keras.metrics.BinaryAccuracy

```python
tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy', dtype=None, threshold=0.5
)
```

For more information, see [tf.keras.metrics.BinaryAccuracy](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/BinaryAccuracy).

## tf.keras.metrics.CategoricalAccuracy

```python
tf.keras.metrics.CategoricalAccuracy(
    name='categorical_accuracy', dtype=None
)
```

For more information, see [tf.keras.metrics.CategoricalAccuracy](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/CategoricalAccuracy).

## tf.keras.metrics.SparseCategoricalAccuracy

```python
tf.keras.metrics.SparseCategoricalAccuracy(
    name='sparse_categorical_accuracy', dtype=None
)
```

For more information, see [tf.keras.metrics.SparseCategoricalAccuracy](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy).

## mindspore.train.Accuracy

```python
mindspore.train.Accuracy(eval_type="classification")
```

For more information, see [mindspore.train.Accuracy](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Accuracy.html#mindspore.train.Accuracy).

## Usage

Accuracy is usually used to calculate accuracy in binary and multiclassification scenarios. MindSpore provides an interface for this function as `mindspore.train.Accuracy`, while TensorFlow provides more options with the same overall functionality and computational logic, but the interface has slightly different format requirements for input data.

In the two frameworks, the evaluation function interface computes and updates the accuracy of the current batch sample using the `update` method and the `update_state` method, which require passing in y_pred and y_true, respectively. Assuming that the current batch data is N and the category is C, the interfaces are roughly distinguished as follows:

MindSpore: `mindspore.train.Accuracy` supports both regular single-label scenarios and multi-label scenarios, with the distinction between 'classification' and 'multilabel' set via the interface parameter `eval_type`. In the `update` method of mindspore, y_pred defaults to probability values in the range [0,1], with a shape of (N,C); y_true needs to be distinguished according to the scenario: For single label, the shape of y can be (N,C) and (N,), and for multi-label, the shape needs to be (N,C). Please refer to the API notes on the official website for details.

TensorFlow version 1.15 provides numerous interfaces for computing Accuracy, and none of these interfaces in this version support multi-label scenarios. The following interfaces can all be configured with sample weights via `sample_weight`, with the following general differences:

- `tf.keras.metrics.Accuracy`: Both y_true and y_pred are category labels (int) by default, and can be used in multi-category cases.

- `tf.keras.metrics.BinaryAccuracy`: y_true defaults to the category label (int), y_pred defaults to the prediction probability, and the threshold is set by the entry threshold, which is usually used in the binary-category case.

- `tf.keras.metrics.CategoricalAccuracy`: y_true defaults to the onehot encoding of the category, and y_pred defaults to the predicted probability, which is usually used in multi-category cases.

- `tf.keras.metrics.SparseCategoricalAccuracy`: y_true defaults to the category label (int), and y_pred defaults to the predicted probability, which is usually used in multi-category cases.

## Code Example

**TensorFlow**:

tf.keras.metrics.Accuracy:

```python
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

y_pred = [1, 2, 3, 4]
y_true = [0, 2, 3, 4]

acc = tf.keras.metrics.Accuracy()
acc.update_state(y_true, y_pred)
print(acc.result().numpy())
# out:0.75
```

tf.keras.metrics.BinaryAccuracy:

```python
y_true = [1, 1, 0, 0]
y_pred = [0.98, 0.7, 0, 0.6]

acc = tf.keras.metrics.BinaryAccuracy()
acc.update_state(y_true, y_pred)
print(acc.result().numpy())
# out: 0.75
```

tf.keras.metrics.CategoricalAccuracy

```python
y_pred = [[0.2, 0.5, 0.4], [0.3, 0.1, 0.4], [0.9, 0.6, 0.4]]
y_true = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
acc = tf.keras.metrics.CategoricalAccuracy()
acc.update_state(y_true, y_pred)
print(acc.result().numpy())
# 0.6666667
```

tf.keras.metrics.SparseCategoricalAccuracy:

```python
y_pred = [[0.2, 0.5, 0.4], [0.3, 0.1, 0.4], [0.9, 0.6, 0.4]]
y_true = [1, 2, 1]
acc = tf.keras.metrics.SparseCategoricalAccuracy()
acc.update_state(y_true, y_pred)
print(acc.result().numpy())
# 0.6666667
```

**MindSpore**:

mindspore.train.Accuracy:

```python
import numpy as np
from mindspore.train import Accuracy
import mindspore as ms

# classification
y_pred = ms.Tensor(np.array([[0.2, 0.5, 0.4], [0.3, 0.1, 0.4], [0.9, 0.6, 0.4]]), ms.float32) # 1 2 0
y_true1 = ms.Tensor(np.array([1, 2, 1]), ms.float32) # y_true1: index of category
y_true2 = ms.Tensor(np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]), ms.float32) # y_true2: one hot encoding

acc = Accuracy('classification')
acc.update(y_pred, y_true1)
print(acc.eval())
# 0.6666666666666666

acc.clear()
acc.update(y_pred, y_true2)
print(acc.eval())
# 0.6666666666666666

# multilabel:
y_pred = ms.Tensor(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]]), ms.float32)
y_true = ms.Tensor(np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]]), ms.float32)

acc = Accuracy('multilabel')
acc.update(y_pred, y_true)
print(acc.eval())
# 0.3333333333333333
```

