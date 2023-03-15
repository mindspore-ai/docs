# 比较与tf.random.uniform_candidate_sampler的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/uniform_candidate_sampler.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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
)  -> Tuple(sampled_candidates, true_expected_count, sampled_expected_count)
```

更多内容详见[tf.random.uniform_candidate_sampler](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/random/uniform_candidate_sampler)。

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
) -> Tuple(sampled_candidates, true_expected_count, sampled_expected_count)
```

更多内容详见[mindspore.ops.uniform_candidate_sampler](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.uniform_candidate_sampler.html)。

## 差异对比

TensorFlow：使用均匀分布对一组内别进行采样，返回三个Tensor。

MindSpore：MindSpore此API实现功能与TensorFlow一致，部分参数名不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | true_classes | true_classes         | -   |
|  | 参数2 | num_true       | num_true          | - |
|  | 参数3 | num_sampled       | num_sampled         | - |
|  | 参数4 | unique       | unique          | - |
|  | 参数5 | range_max       | range_max         | - |
|  | 参数6 | seed       | seed          | - |
| | 参数7 | - | remove_accidental_hits      | 表示是否移除accidental hit。默认值：False|
| | 参数8 | name | -           | 不涉及 |
|返回参数| 参数1 | sampled_candidates |   sampled_candidates        |- |
| | 参数2 | true_expected_count |     true_expected_count    | - |
| |  参数3| sampled_expected_count |     sampled_expected_count     | - |

### 代码示例1

MindSpore和TensorFlow输出结果一致。

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
