# 比较与tf.keras.metrics.Accuracy、tf.keras.metrics.BinaryAccuracy、tf.keras.metrics.CategoricalAccuracy、tf.keras.metrics.SparseCategoricalAccuracy的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/metricAcc.md " target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## tf.keras.metrics.Accuracy

```python
tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)
```

更多内容详见[tf.keras.metrics.Accuracy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/Accuracy)。

```python
tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy', dtype=None, threshold=0.5
)
```

更多内容详见[tf.keras.metrics.BinaryAccuracy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/BinaryAccuracy)。

```python
tf.keras.metrics.CategoricalAccuracy(
    name='categorical_accuracy', dtype=None
)
```

更多内容详见[tf.keras.metrics.CategoricalAccuracy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/CategoricalAccuracy)。

```python
tf.keras.metrics.SparseCategoricalAccuracy(
    name='sparse_categorical_accuracy', dtype=None
)
```

更多内容详见[tf.keras.metrics.SparseCategoricalAccuracy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy)。

## mindspore.nn.Accuracy

```python
mindspore.nn.Accuracy(eval_type="classification")
```

更多内容详见[mindspore.nn.Accuracy](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Accuracy.html#mindspore.nn.Accuracy)。

## 使用方式

Accuracy通常用于二分类和多分类场景下准确率的计算。MindSpore提供相关功能的接口为`mindspore.nn.Accuracy`；TensorFlow提供了较多选择，整体功能及计算逻辑相同，但接口对输入数据的格式要求略有不同。

在两框架中，评估函数接口分别使用`update`方法和`update_state`方法对当前batch样本的准确率进行计算和更新，需要传入y_pred和y_true，假设当前batch数据为N，类别为C，接口大致区别如下：

MindSpore：`mindspore.nn.Accuracy`支持常规单标签场景和多标签场景，通过接口入参`eval_type`设置'classification'和'multilabel'区分。mindspore的`update`方法中，y_pred默认为[0,1]范围内的概率值， shape为(N,C) ；y_true需要依据场景区分：单标签时，y的shape可以为 (N,C) 和(N,)，多标签时，需要shape为 (N,C)，详情请参考官网API注释。

TensorFlow1.15版本提供了众多计算Accuracy的接口，且此版本的这些接口都不支持多标签场景。下述接口都可以通过`sample_weight`配置样本权重，大致区别如下：

- `tf.keras.metrics.Accuracy`：y_true和y_pred默认都为类别标签(int)，可用于多分类情况。

- `tf.keras.metrics.BinaryAccuracy`：y_true默认为类别标签(int)，y_pred默认为预测概率，通过入参threshold设置阈值，通常用于二分类情况。

- `tf.keras.metrics.CategoricalAccuracy`：y_true默认为类别的onehot编码，y_pred默认为预测概率，通常用于多分类情况。

- `tf.keras.metrics.SparseCategoricalAccuracy`：y_true默认为类别标签(int)，y_pred默认为预测概率，通常用于多分类情况。

## 代码示例

**TensorFlow**：

 tf.keras.metrics.Accuracy：

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

tf.keras.metrics.BinaryAccuracy：

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

tf.keras.metrics.SparseCategoricalAccuracy：

```python
y_pred = [[0.2, 0.5, 0.4], [0.3, 0.1, 0.4], [0.9, 0.6, 0.4]]
y_true = [1, 2, 1]
acc = tf.keras.metrics.SparseCategoricalAccuracy()
acc.update_state(y_true, y_pred)
print(acc.result().numpy())
# 0.6666667
```

**MindSpore**：

mindspore.nn.Accuracy：

```python
import numpy as np
from mindspore import nn, Tensor
import mindspore

# classification
y_pred = Tensor(np.array([[0.2, 0.5, 0.4], [0.3, 0.1, 0.4], [0.9, 0.6, 0.4]]), mindspore.float32) # 1 2 0
y_true1 = Tensor(np.array([1, 2, 1]), mindspore.float32) # y_true1: index of category
y_true2 = Tensor(np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]), mindspore.float32) # y_true2: one hot encoding

acc = nn.Accuracy('classification')
acc.update(y_pred, y_true1)
print(acc.eval())
# 0.6666666666666666

acc.clear()
acc.update(y_pred, y_true2)
print(acc.eval())
# 0.6666666666666666

# multilabel:
y_pred = Tensor(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]]), mindspore.float32)
y_true = Tensor(np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]]), mindspore.float32)

acc = nn.Accuracy('multilabel')
acc.update(y_pred, y_true)
print(acc.eval())
# 0.3333333333333333
```
