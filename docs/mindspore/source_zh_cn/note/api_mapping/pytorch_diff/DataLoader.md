# 比较与torch.utils.data.DataLoader的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/DataLoader.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.utils.data.DataLoader

```python
class torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
    num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
    timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

更多内容详见[torch.utils.data.DataLoader](https://pytorch.org/docs/1.5.0/data.html#torch.utils.data.DataLoader)。

## MindSpore

MindSpore中的数据集无需加载器。

## 使用方式

PyTorch：需要将数据集和采样器对象，及分批、混洗、并行等参数传入此接口，以实现具有采样、分批、混洗等功能的并行数据迭代。

MindSpore：采样器对象及混洗、并行等参数可以在创建数据集对象时传入，也可以通过数据集类内方法定义，则数据集对象直接具有采样、分批、混洗、并行数据迭代等功能，无需额外的加载器。

## 代码示例

```python
import torch
import mindspore.dataset as ds
import numpy as np

np.random.seed(0)
data = np.random.random((6, 2, 2))

# The following implements data iteration with PyTorch.
dataset = torch.utils.data.TensorDataset(torch.Tensor(data))
sampler = torch.utils.data.SequentialSampler(dataset)
loader = torch.utils.data.DataLoader(dataset, batch_size=3)
print(next(iter(loader)))
# Out:
# [tensor([[[0.5488, 0.7152],
#          [0.6028, 0.5449]],
#
#         [[0.4237, 0.6459],
#          [0.4376, 0.8918]],
#
#         [[0.9637, 0.3834],
#          [0.7917, 0.5289]]])]

# The following implements data iteration with MindSpore.
sampler = ds.SequentialSampler()
dataset = ds.NumpySlicesDataset(data, sampler=sampler)
dataset = dataset.batch(batch_size=3)
iterator = dataset.create_dict_iterator()
print(next(iter(iterator)))
# Out:
# {'column_0': Tensor(shape=[3, 2, 2], dtype=Float64, value=
# [[[ 5.48813504e-01,  7.15189366e-01],
#   [ 6.02763376e-01,  5.44883183e-01]],
#  [[ 4.23654799e-01,  6.45894113e-01],
#   [ 4.37587211e-01,  8.91773001e-01]],
#  [[ 9.63662761e-01,  3.83441519e-01],
#   [ 7.91725038e-01,  5.28894920e-01]]])}
```
