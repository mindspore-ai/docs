# 比较与torch.utils.data.SubsetRandomSampler的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SubsetRandomSampler.md)

## torch.utils.data.SubsetRandomSampler

```python
class torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

更多内容详见[torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.SubsetRandomSampler)。

## mindspore.dataset.SubsetRandomSampler

```python
class mindspore.dataset.SubsetRandomSampler(indices, num_samples=None)
```

更多内容详见[mindspore.dataset.SubsetRandomSampler](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.SubsetRandomSampler.html)。

## 差异对比

PyTorch：给定样本的索引序列，从序列中随机获取索引对数据集进行采样，支持指定采样逻辑。

MindSpore：给定样本的索引序列，从序列中随机获取索引对数据集进行采样，不支持指定采样逻辑。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | indices | indices    | - |
|     | 参数2 | generator  | -      | 指定额外的采样逻辑，MindSpore为全局随机采样 |
|     | 参数3 | -   | num_samples  | 指定采样器返回的样本数量 |

## 代码示例

```python
import torch
from torch.utils.data import SubsetRandomSampler

torch.manual_seed(0)

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = SubsetRandomSampler(indices=[0, 2])
dataloader = torch.utils.data.DataLoader(ds, sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# tensor([2])
# tensor([0])
```

```python
import mindspore as ms
from mindspore.dataset import SubsetRandomSampler

ms.dataset.config.set_seed(1)

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = SubsetRandomSampler(indices=[0, 2])
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# [Tensor(shape=[], dtype=Int64, value= 2)]
# [Tensor(shape=[], dtype=Int64, value= 0)]
```
