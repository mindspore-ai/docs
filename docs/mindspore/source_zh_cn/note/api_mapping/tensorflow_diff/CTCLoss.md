# 比较与tf.nn.ctc_loss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/CTCLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.nn.ctc_loss

```text
tf.nn.ctc_loss(
    labels,
    logits,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=None,
    name=None
) -> Tensor
```

更多内容详见[tf.nn.ctc_loss](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/ctc_loss)。

## mindspore.ops.CTCLoss

```text
class mindspore.ops.CTCLoss(
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False
)(x, labels_indices, labels_values, sequence_length) -> (Tensor, Tensor)
```

更多内容详见[mindspore.ops.CTCLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.CTCLoss.html)。

## 差异对比

TensorFlow：计算连续时间序列和目标序列之间的损失。并且只返回一个和`label_length`形状一致的`loss`。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，返回`loss`及其`梯度`。

| 分类 | 子类   | TensorFlow      | MindSpore    | 差异          |
| ---- | ----- | --------------------------------- | ------ | ------ |
|      | 参数1  | labels | labels_values  | 功能一致，参数名称不同，但是在MindSpore中秩必须为1 |
|      | 参数2  | logits  | x    | 功能一致，参数名称不同    |
|      | 参数3  | label_length  | sequence_length  | 功能一致，参数名称不同    |
|      | 参数6  | logit_length  |    -   | MindSpore无此参数     |
|      | 参数6  | logits_time_major  |    -   |  控制logits的排布方式，MindSpore无此参数     |
|      | 参数8  | unique   |     -      | MindSpore无此参数     |
|      | 参数7  | blank_index  |    -    | MindSpore无此参数，为-1时，blank用num_classes-1表示，此时与MindSpore一致 |
|      | 参数9  | name     |     -     | 不涉及                                |
|      | 参数1  | -      | preprocess_collapse_repeated | 在CTC计算之前将折叠重复标签，TensorFlow无此参数     |
|      | 参数2  | -             | ctc_merge_repeated           | 是否合并非空白标签，TensorFlow无此参数  |
|      | 参数11 | - |     ignore_longer_outputs_than_inputs   | 是否忽略输出比输入长的序列，TensorFlow无此参数      |
|      | 参数10 |    -   | labels_indices   | labels_indices[i, :] = [b, t] 表示 labels_values[i] 存储 (batch b, time t) 的ID，保证了labels_values的秩为1 |

### 代码示例

> 功能一致

```python
# TensorFlow
import tensorflow as tf
import numpy as np

label = tf.Variable([[0],
                     [1]])
logits = tf.constant(np.array([[[0.56352055, -0.24474338, -0.29601783],[0.8030011, -1.2187808, -0.6991761]],[[-0.81990826, -0.3598757, 0.50144005],[-1.0980303, 0.60394925, 0.3771529]]]), dtype=tf.float32)
label_length = tf.Variable(np.array([1, 1]))
logits_length = tf.Variable(np.array([2, 2]))
loss = tf.nn.ctc_loss(label, logits, label_length=label_length, logit_length=logits_length)
print(loss.numpy().shape)
# (2,)

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np

x = Tensor(np.array([[[0.56352055, -0.24474338, -0.29601783], [0.8030011, -1.2187808, -0.6991761]], [[-0.81990826, -0.3598757, 0.50144005], [-1.0980303, 0.60394925, 0.3771529]]]).astype(np.float32))

labels_indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int64)
labels_values = Tensor(np.array([0, 1]), mindspore.int32)
sequence_length = Tensor(np.array([2, 2]), mindspore.int32)

ctc_loss = ops.CTCLoss()
loss, gradient = ctc_loss(x, labels_indices, labels_values, sequence_length)
print(loss.shape)
# (2,)
```
