# 比较与tf.keras.metrics.Recall的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/metricRecall.md)

## tf.keras.metrics.Recall

```python
tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
```

更多内容详见[tf.keras.metrics.Recall](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/Recall)。

## mindspore.nn.Recall

```python
mindspore.nn.Recall(eval_type="classification")
```

更多内容详见[mindspore.nn.Recall](https://mindspore.cn/docs/zh-CN/r1.7/api_python/nn/mindspore.nn.Recall.html#mindspore.nn.Recall)。

## 使用方式

计算公式相同，都为Recall = true_positives / (true_positives + false_negatives)，但对输入处理逻辑不同，假设输入真实值为y_true，预测值为y_pred,差异如下：

TensorFlow：TensorFlow1.15版本此接口仅支持二分类的单标签场景，最终返回recall的均值。y_true被映射到布尔型变量(除0外都被映射为1)；y_pred>`thresholds`被认为预测为正类，y_pred<=`thresholds`被认为预测为负类。入参大致情况如下：

- `thresholds`：设置预测值判别为正确的阈值，默认值为None，此时会使用0.5对预测值进行正确与否的判断。

- `top_k`：默认值为None，此时使用全量样本。设置后thresholds会失效，此时选用预测值topk的样本进行计算。

- `class_id`：指定类别id进行单类别计算。

MindSpore：支持单标签和多标签场景，通过布尔型入参`average`控制返回结果，默认为False，返回各类别的recall统计值，若设置为True，则返回均值。

以下述代码为例，两接口的输入为相同样例，默认参数情况下，`mindspore.nn.Recall`将输入解析为5条样本的3分类问题，输出长度为3的列表表示各类的recall，`tf.keras.metrics.Recall`将输入解析为15条样本的2分类问题，输出所有样本的recall均值。

## 代码示例

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
from mindspore import nn, Tensor

x = Tensor(np.array([[0.2, 0.5, 0.1], [0.3, 0.1, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.97], [0.2, 0.6, 0.8]]))
y = Tensor(np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))
metric = nn.Recall('classification')
metric.update(x, y)
print(metric.eval(average=True), metric.eval())

# output: 0.8333333333333334 [1. 0.5 1. ]
```
