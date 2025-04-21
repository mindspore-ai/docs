# 比较与torch.utils.data.RandomSampler的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomSampler.md)

## torch.utils.data.RandomSampler

```python
class torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

更多内容详见[torch.utils.data.RandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.RandomSampler)。

## mindspore.dataset.RandomSampler

```python
class mindspore.dataset.RandomSampler(replacement=False, num_samples=None)
```

更多内容详见[mindspore.dataset.RandomSampler](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.RandomSampler.html)。

## 差异对比

PyTorch：随机采样器，支持指定采样逻辑。

MindSpore：随机采样器，不支持指定采样逻辑。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | data_source | -  | 被采样的数据集对象，MindSpore不需要传入 |
|     | 参数2 | replacement   | replacement |- |
|     | 参数3 | num_samples   | num_samples  |- |
|     | 参数4 | generator  | -  | 指定额外的采样逻辑，MindSpore为全局随机采样 |

### 代码示例

```python
import torch
from torch.utils.data import RandomSampler

torch.manual_seed(1)

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = RandomSampler(ds, num_samples=2, replacement=True)
dataloader = torch.utils.data.DataLoader(ds, sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# tensor([2])
# tensor([0])
```

```python
import mindspore as ms
from mindspore.dataset import RandomSampler

ms.dataset.config.set_seed(3)

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = RandomSampler(num_samples=2, replacement=True)
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# [Tensor(shape=[], dtype=Int64, value= 2)]
# [Tensor(shape=[], dtype=Int64, value= 0)]
```
