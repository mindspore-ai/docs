# Function Differences with tf.keras.metrics.AUC

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/metricAUC.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.metrics.AUC

```python
tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
    dtype=None, thresholds=None
)
```

For more information, see [tf.keras.metrics.AUC](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/metrics/AUC).

## mindspore.train.auc

```python
mindspore.train.auc(x, y, reorder=False)
```

For more information, see [mindspore.train.auc](https://mindspore.cn/docs/en/master/api_python/train/mindspore.train.auc.html#mindspore.train.auc).

## Usage

TensorFlow: Input y_pred and y_true, and control whether the return value is based on the ROC curve or the Precision-Recall curve via the input `curve`. In addition, users can set their own parameters such as the number of thresholds `num_thresholds` and the threshold value `thresholds`. Support `interpolate_pr_auc()` method (there is no corresponding function in MindSpore). Please check the API interface for details of implementation and usage. TensorFlow version 1.15 only supports binary classification.

MindSpore: Before calling the `mindspore.nn.auc` interface, FPR(false positive rate) and TPR(true positive rate) should be derived using `mindspore.nn.ROC`, and the threshold value is determined by the y_pred element value size during calculation. The computed FPR and TPR are passed into `mindspore.nn.auc` for AUC calculation. Binary classification and multiclassification are supported.

## Code Example

```python
from mindspore.train import ROC, auc
import numpy as np

x = ms.Tensor(np.array([[0.28, 0.55, 0.15, 0.05], [0.10, 0.20, 0.05, 0.05], [0.20, 0.05, 0.15, 0.05],
                    [0.05, 0.05, 0.05, 0.75], [0.05, 0.05, 0.05, 0.75]]))
y = ms.Tensor(np.array([0, 1, 2, 3, 2]))
metric = ROC(class_num=4)
metric.update(x, y)
fpr, tpr, thresholds = metric.eval()
print(fpr)
# out: [array([0.        , 0.33333333, 0.33333333, 0.66666667, 1.        ]), array([0.        , 0.33333333, 1.        ]),
# array([0.        , 0.33333333, 1.        ]), array([0.        , 0.33333333, 1.        ])]

print(tpr)
# out: [array([0., 0., 1., 1., 1.]), array([0., 0., 1.]), array([0., 0., 1.]), array([0., 0., 1.])]

# calculate auc for class 0
output = auc(fpr[0], tpr[0])
print(output)
# out: 0.6666666666666667


import tensorflow as tf
tf.enable_eager_execution()

m = tf.keras.metrics.AUC(num_thresholds=3)
m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
print(m.result().numpy())
# out: 0.75
```
