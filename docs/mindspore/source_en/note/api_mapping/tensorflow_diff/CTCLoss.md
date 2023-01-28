# Function Differences with tf.nn.ctc_loss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/CTCLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.nn.ctc_loss](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/nn/ctc_loss).

## mindspore.ops.CTCLoss

```text
class mindspore.ops.CTCLoss(
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False
)(x, labels_indices, labels_values, sequence_length) -> (Tensor, Tensor)
```

For more information, see [mindspore.ops.CTCLoss](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.CTCLoss.html).

## Differences

TensorFlow: Calculate the loss between the continuous time sequence and the target sequence. And return only a `loss` that is consistent with the shape of `label_length`.

MindSpore: MindSpore API basically implements the same function as TensorFlow, returns `loss` and its `gradient`.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|   Parameters   | Parameter 1  | labels | labels_values  | The function is the same, and the parameter name is different, but in MindSpore the rank must be 1 |
|      | Parameter 2  | logits  | x    | Same function, different parameter names    |
|      | Parameter 3  | label_length  | sequence_length  | Same function, different parameter names    |
|      | Parameter 4  | logit_length  |    -   | MindSpore does not have this parameter     |
|      | Parameter 5  | logits_time_major  |    -   |  Control how logits are arranged. MindSpore does not have this parameter     |
|      | Parameter 6  | unique   |     -      | MindSpore does not have this parameter     |
|      | Parameter 7  | blank_index  |    -    | MindSpore does not have this parameter. When it is -1, blank is represented by num_classes-1, which is consistent with MindSpore at this time |
|      | Parameter 8  | name     |     -     | Not involved                                |
|      | Parameter 9  | -      | preprocess_collapse_repeated | Collapse duplicate labels before CTC calculation. TensorFlow does not have this parameter     |
|      | Parameter 10  | -             | ctc_merge_repeated           | Whether to merge non-blank labels. TensorFlow does not have this parameter  |
|      | Parameter 11 | - |     ignore_longer_outputs_than_inputs   | Whether to ignore sequences whose output is longer than the input, TensorFlow does not have this parameter      |
|      | Parameter 12 |    -   | labels_indices   | labels_indices[i, :] = [b, t] means that labels_values[i] stores the IDs of (batch b, time t), which guarantees that the rank of labels_values is 1 |

### Code Example

> Consistent functions.

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
