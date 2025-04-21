# 比较与torch.utils.data.WeightedRandomSampler的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/WeightedRandomSampler.md)

## torch.utils.data.WeightedRandomSampler

```python
class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
```

更多内容详见[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.WeightedRandomSampler)。

## mindspore.dataset.WeightedRandomSampler

```python
class mindspore.dataset.WeightedRandomSampler(weights, num_samples=None, replacement=True)
```

更多内容详见[mindspore.dataset.WeightedRandomSampler](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.WeightedRandomSampler.html)。

## 差异对比

PyTorch：给定样本的权重列表，根据权重的大小对样本进行采样，支持指定采样逻辑。

MindSpore：给定样本的权重列表，根据权重的大小对样本进行采样，不支持指定采样逻辑。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | weights  | weights     | - |
|     | 参数2 | num_samples   |num_samples   | - |
|     | 参数3 | replacement  | replacement   | - |
|     | 参数4 | generator   | -  | 指定额外的采样逻辑，MindSpore为全局随机采样 |

## 代码示例

```python
import torch
from torch.utils.data import WeightedRandomSampler

torch.manual_seed(0)

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(1, 5)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = WeightedRandomSampler(weights=[0.1, 0.1, 0.9, 0.9], num_samples=4)
dataloader = torch.utils.data.DataLoader(ds, sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# tensor([4])
# tensor([3])
# tensor([4])
# tensor([4])
```

```python
import mindspore as ms
from mindspore.dataset import WeightedRandomSampler

ms.dataset.config.set_seed(4)

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(1, 5)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = WeightedRandomSampler(weights=[0.1, 0.1, 0.9, 0.9], num_samples=4)
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# [Tensor(shape=[], dtype=Int64, value= 4)]
# [Tensor(shape=[], dtype=Int64, value= 3)]
# [Tensor(shape=[], dtype=Int64, value= 4)]
# [Tensor(shape=[], dtype=Int64, value= 4)]
```
