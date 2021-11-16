# 比较与tf.keras.metrics.AUC的功能差异

## tf.keras.metrics.AUC

```python
tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
    dtype=None, thresholds=None
)
```

更多内容详见[tf.keras.metrics.AUC](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/AUC)。

## mindspore.nn.auc

```python
mindspore.nn.auc(x, y, reorder=False)
```

更多内容详见[mindspore.nn.auc](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.auc.html#mindspore.nn.auc)。

## 使用方式

TensorFlow：输入y_pred和y_true，通过入参`curve`控制返回值是基于ROC曲线还是Precision-Recall曲线。此外，用户可自行设置阈值数目`num_thresholds`，阈值`thresholds`等参数。支持`interpolate_pr_auc()`方法(MindSpore中暂无此对应功能)，具体实现及使用方法请查看API接口详情。TensorFlow1.15版本此接口只支持二分类。

MindSpore：调用`mindspore.nn.auc`接口前需先使用`mindspore.nn.ROC`得出FPR(false positive rate)和TPR(true positive rate)，计算时阈值由y_pred元素值大小决定。计算得到的FPR和TPR传入`mindspore.nn.auc`进行AUC的计算。支持二分类和多分类。

## 代码示例

```python
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[0.28, 0.55, 0.15, 0.05], [0.10, 0.20, 0.05, 0.05], [0.20, 0.05, 0.15, 0.05],
                    [0.05, 0.05, 0.05, 0.75], [0.05, 0.05, 0.05, 0.75]]))
y = Tensor(np.array([0, 1, 2, 3, 2]))
metric = nn.ROC(class_num=4)
metric.update(x, y)
fpr, tpr, thresholds = metric.eval()
print(fpr)
# out: [array([0.        , 0.33333333, 0.33333333, 0.66666667, 1.        ]), array([0.        , 0.33333333, 1.        ]),
# array([0.        , 0.33333333, 1.        ]), array([0.        , 0.33333333, 1.        ])]

print(tpr)
# out: [array([0., 0., 1., 1., 1.]), array([0., 0., 1.]), array([0., 0., 1.]), array([0., 0., 1.])]

# calculate auc for class 0
output = nn.auc(fpr[0], tpr[0])
print(output)
# out: 0.6666666666666667


import tensorflow as tf
tf.enable_eager_execution()

m = tf.keras.metrics.AUC(num_thresholds=3)
m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
print(m.result().numpy())
# out: 0.75
```
