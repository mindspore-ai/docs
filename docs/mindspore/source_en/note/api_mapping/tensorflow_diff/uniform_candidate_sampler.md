# Function Differences with tf.random.uniform_candidate_sampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/uniform_candidate_sampler.md)

## tf.random.uniform_candidate_sampler

```text
tf.random.uniform_candidate_sampler(
    true_classes,
    num_true,
    num_sampled,
    unique,
    range_max,
    seed=None,
    name=None
)(sampled_candidates, true_expected_count, sampled_expected_count)  -> Tuple
```

For more information, see [tf.random.uniform_candidate_sampler](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/random/uniform_candidate_sampler).

## mindspore.ops.uniform_candidate_sampler

```text
mindspore.ops.uniform_candidate_sampler(
    true_classes,
    num_true,
    num_sampled,
    unique,
    range_max,
    seed=0,
    remove_accidental_hits=False
)(sampled_candidates, true_expected_count, sampled_expected_count) -> Tuple
```

For more information, see [mindspore.ops.uniform_candidate_sampler](https://www.mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.uniform_candidate_sampler.html).

## Differences

TensorFlow: Sample a set of internal aliases using a uniform distribution and return three Tensors.

MindSpore: MindSpore API implements the same functions as TensorFlow, with some parameter names different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | true_classes | true_classes         | -   |
|  | Parameter 2 | num_true       | num_true          | - |
|  | Parameter 3 | num_sampled       | num_sampled         | - |
|  | Parameter 4 | unique       | unique          | - |
|  | Parameter 5 | range_max       | range_max         | - |
|  | Parameter 6 | seed       | seed          | - |
| | Parameter 7 | - | remove_accidental_hits      | Indicates whether to remove the accidental hit. Default: False|
| | Parameter 8 | name | -           | Not involved |
| Return Parameters | Parameter 1 | sampled_candidates |   sampled_candidates        |-|
| | Parameter 2 | true_expected_count |     true_expected_count    | - |
| | Parameter 3| sampled_expected_count |     sampled_expected_count     | - |

### Code Example 1

The outputs of MindSpore and TensorFlow are consistent.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

data = tf.constant(np.random.rand(5, 3), dtype=tf.int64)
out1, out2, out3 = tf.random.uniform_candidate_sampler(data, 3, 3, False, 5, 0)
print(out1.shape)
# (3,)
print(out2.shape)
# (5, 3)
print(out3.shape)
# (3,)

# MindSpore
import mindspore
import numpy as np
from mindspore.ops import function as ops
from mindspore import Tensor

data = Tensor(np.random.rand(5, 3), mindspore.int64)
out1, out2, out3 = ops.uniform_candidate_sampler(data, 3, 3, False, 5, 0)
print(out1.shape)
# (3,)
print(out2.shape)
# (5, 3)
print(out3.shape)
# (3,)

```
