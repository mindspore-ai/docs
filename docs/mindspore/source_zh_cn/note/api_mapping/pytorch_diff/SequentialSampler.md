# 比较与torch.utils.data.SequentialSampler的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SequentialSampler.md)

## torch.utils.data.SequentialSampler

```python
class torch.utils.data.SequentialSampler(data_source)
```

更多内容详见[torch.utils.data.SequentialSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.SequentialSampler)。

## mindspore.dataset.SequentialSampler

```python
class mindspore.dataset.SequentialSampler(start_index=None, num_samples=None)
```

更多内容详见[mindspore.dataset.SequentialSampler](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.SequentialSampler.html)。

## 差异对比

PyTorch：按数据集的顺序采样数据集样本。

MindSpore：按数据集的顺序采样数据集样本，支持指定顺序索引和样本数量。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | data_source | -  | 被采样的数据集对象，MindSpore不需要传入 |
|     | 参数2 | -   | start_index  | 采样的起始样本索引 |
|     | 参数3 | -   | num_samples  | 指定采样器返回的样本数量 |

## 代码示例

```python
import torch
from torch.utils.data import SequentialSampler

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = SequentialSampler(ds)
dataloader = torch.utils.data.DataLoader(ds, sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# tensor([0])
# tensor([1])
# tensor([2])
# tensor([3])
```

```python
import mindspore as ms
from mindspore.dataset import SequentialSampler

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = SequentialSampler()
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# [Tensor(shape=[], dtype=Int64, value= 0)]
# [Tensor(shape=[], dtype=Int64, value= 1)]
# [Tensor(shape=[], dtype=Int64, value= 2)]
# [Tensor(shape=[], dtype=Int64, value= 3)]
```
