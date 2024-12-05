# 动态shape相关迁移策略

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindspore/source_zh_cn/migration_guide/dynamic_shape.md)

想要了解动态shape，需要先了解什么是静态shape。
静态shape指在网络执行阶段Tensor的shape没有发生变化。
比如resnet50网络如果保证图片的输入shape一直是`224*224`的，那么在网络训练阶段，四个残差模块的输出Tensor的shape分别是`B*64*56*56`，`B*128*28*28`，`B*256*14*14`，`B*512*7*7`，`B`指`BatchSize`，在训练过程中也是固定的，此时网络中全部是静态的shape，没有动态shape。
如果输入的shape不一定是`224*224`的，那么四个残差模块输出Tensor的shape将会随输入shape变化，此时就不是静态shape，而是动态shape了。一般动态shape引入的原因和解决办法有：

## 输入shape不固定

比如输入图片需要有不同的shape，音频的label需要不同长度，这都会引入动态shape；

这种场景可以读代码分析数据处理的输出shape是否固定，也可以直接打印数据处理输出的shape，进行对比：

```python
for batch_idx, (data, target) in enumerate(data_loader):
    print(batch_idx, data.shape, target.shape)
    print("="*20)
```

**解决办法**

可通过mask机制把动态shape转换成静态shape，mask机制示例代码如下：

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

## 网络执行过程中有引发shape变化的API

在网络执行过程中可能有一些操作会引起Tensor的shape变化。

引起这种场景常见的API有：

| API | 功能描述 | 引发动态shape场景 |
| ---- | ----- | ------- |
| StridedSlice/Slice | 切片，用户编程时也可以使用 [start_idx:end_idx]这种方式 | 当切片下标是变量时 |
| TopK | 取前K大 | 当K取值不定时 |
| Gather | 取Tensor在指定 axis 上索引对应的元素组成的切片 | 当index长度不定时 |
| UnsortedSegmentX | 包含UnsortedSegmentSum，UnsortedSegmentMax等沿分段计算输入Tensor的某个计算 | 当分段不固定时 |
| Sampler | 取样器相关操作，比如where，random.choice等 | 当抽取数量不固定时 |
| ReduceX | ReduceSum，ReduceMean等归约操作 | 当axis不固定时 |
| Transpose | 根据轴进行变换 | 当变化轴不定时 |
| Unique | 去重 | 使用就会引入动态shape |
| MaskedSelect | 根据bool型的mask取值 | 使用就会引入动态shape |
| NonZero | 计算非零元素的下标 | 使用就会引入动态shape |

比如：

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

在网络训练时有个切片的操作`x[:k]`这里的k不是一个常量，会导致`x[:k]`的shape随k的值改变，导致后续所有和`x[:k]`相关的操作的shape不确定。

**解决办法**

如果遇到该场景引入动态shape，本质是需要将动态变化的值修改为固定的shape来解决问题。
如TopK算子，若执行过程中K是变化的，则会引入动态shape。
可先固定一个最大目标数，先按静态shape获取所有目标的置信度，再选择K个最高的目标作为结果输出，其他目标通过mask机制去除。示例代码如[FasterRCNN](https://gitee.com/mindspore/models/blob/master/official/cv/FasterRCNN/src/FasterRcnn/faster_rcnn.py)的multiclass_nms接口。

## 控制流不同分支引入shape上的变化

网络中可能会有一些控制流的输出是不一样的，而当控制流的条件控制项不是固定的时，可能会引发动态shape，比如：

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

在这个过程其实有两个地方有动态shape，一个是`cond=True`时`masked_select`结果的shape是动态，另外是控制流，由于cond不定，控制流两个分支的shape输出不同也会造成动态shape。

**解决办法**

可尝试用equal、select算子替换if条件，示例代码如下：

```python
# 引入控制流的代码示例：
if ms.ops.reduce_sum(object_masks)==0:
    stage2_loss = stage2_loss.fill(0.0)
# 修改后的代码示例：
stage2_loss = ms.ops.select(ms.ops.equal(ms.ops.reduce_sum(object_masks), 0), stage2_loss.fill(0), stage2_loss)
```
