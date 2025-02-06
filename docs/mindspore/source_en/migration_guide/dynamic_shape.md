# Strategies for Migrating Dynamic Shape

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_en/migration_guide/dynamic_shape.md)

To know dynamic shape, you need to know what is a static shape.
Static shape indicates that the shape of a tensor does not change during network execution.
For example, on the ResNet50 network, if the input shape of an image is always `224*224`, the shapes of the output Tesnor of the four residual modules are `B*64*56*56`, `B*128*28*28`, `B*256*14*14`, and `B*512*7*7` respectively in the network training phase. `B` indicates `BatchSize`, which is also fixed during the training. In this case, all shapes on the network are static and no dynamic shape is available.
If the input shape may no+t be `224*224`, the shape of the output tensor of the four residual modules varies with the input shape. In this case, the shape is dynamic instead of static. Generally, dynamic shape is introduced due to the following reasons:

## Input Shape not Fixed

For example, the input image has different shapes, and the audio label has different lengths. In this case, dynamic shapes are introduced.

In this scenario, you can read the code to check whether the output shape of data processing is fixed, or directly print the output shape of data processing for comparison.

```python
for batch_idx, (data, target) in enumerate(data_loader):
    print(batch_idx, data.shape, target.shape)
    print("="*20)
```

**Solution**

Dynamic shape can be converted to static shape through the mask mechanism. Mask mechanism example code is as follows:

```python
def _convert_ids_and_mask(input_tokens, seq_max_bucket_length):
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < seq_max_bucket_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == seq_max_bucket_length
    assert len(input_mask) == seq_max_bucket_length

    return input_ids, input_mask
```

## APIs that Cause Shape Changes During Network Execution

During network execution, some operations may cause tensor shape changes.

The common APIs that cause this scenario are as follows:

| API | Description| Dynamic Shape Scenario|
| ---- | ----- | ------- |
| StridedSlice/Slice | Specifies a slice. You can also use [start_idx:end_idx] during programming.| The slice subscript is a variable.|
| TopK | Obtains the first K data.| The value of K is not fixed.|
| Gather | Obtains the slice consisting of the elements corresponding to the tensor index on the specified axis.| The index length is not fixed.|
| UnsortedSegmentX | Specifies computation of an input tensor, including UnsortedSegmentSum and UnsortedSegmentMax.| The segment is not fixed.|
| Sampler | Specifies sampler-related operations, such as where and random.choice.| The sample quantity is not fixed.|
| ReduceX | Specifies a reduction operation, such as ReduceSum and ReduceMean.| The axis is not fixed.|
| Transpose | Performs transformation based on the axis.| The axis is not fixed.|
| Unique | Deduplicates data.| Dynamic shape is introduced when this API is used.|
| MaskedSelect | Obtains the value of mask based on the boolean type.| Dynamic shape is introduced when this API is used.|
| NonZero | Obtains the positions of all non-zero values.| Dynamic shape is introduced when this API is used.|

For example:

```python
import numpy as np
import mindspore as ms
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
k = ms.Tensor(np.random.randint(1, 10), ms.int64)
print(k)
print(x[:k].shape)
# 6
# (6,)
```

During network training, there is a slicing operation `x[:k]`. Here, k is not a constant. As a result, the shape of `x[:k]` changes with the value of k, and the shape of all subsequent operations related to `x[:k]` is uncertain.

**Solution**

If this scenario is encountered to introduce a dynamic shape, the essence is that the dynamically changing values need to be modified to a fixed shape to solve the problem.
As in the case of the TopK operator, if K is changing during execution, a dynamic shape is introduced.
You can fix a maximum number of targets, first get the confidence level of all targets by static shape, then choose the K number of highest targets as the result output, and other targets are removed by mask mechanism. Sample code such as the multiclass_nms interface of [FasterRCNN](https://gitee.com/mindspore/models/blob/master/official/cv/FasterRCNN/src/FasterRcnn/faster_rcnn.py).

## Shape Changes Introduced by Different Branches of Control Flows

The output of some control flows on the network may be different. When the condition control items of the control flows are not fixed, dynamic shape may be triggered. For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
np.random.seed(1)
x = ms.Tensor(np.random.uniform(0, 1, (10)).astype(np.float32))
cond = (x > 0.5).any()

if cond:
    y = ops.masked_select(x, x > 0.5)
else:
    y = ops.zeros_like(x)
print(x)
print(cond)
print(y)

# [4.17021990e-01 7.20324516e-01 1.14374816e-04 3.02332580e-01
#  1.46755889e-01 9.23385918e-02 1.86260208e-01 3.45560730e-01
#  3.96767467e-01 5.38816750e-01]
# True
# [0.7203245  0.53881675]
```

In this process, there are two dynamic shapes. One is that the shape of the `masked_select` result is dynamic if `cond=True`. The other is the control flow. Because `cond` is uncertain, the shape output of the two branches of the control flow is different, which also causes the dynamic shape.

**Solution**

You can try to use equal, select operators to replace the if condition. Sample code is as follows:

```python
# Code example for introducing control flow:
if ms.ops.reduce_sum(object_masks)==0:
    stage2_loss = stage2_loss.fill(0.0)
# modified code example
stage2_loss = ms.ops.select(ms.ops.equal(ms.ops.reduce_sum(object_masks), 0), stage2_loss.fill(0), stage2_loss)
```
