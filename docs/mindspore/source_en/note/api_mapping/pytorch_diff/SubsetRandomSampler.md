# Differences with torch.utils.data.SubsetRandomSampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SubsetRandomSampler.md)

## torch.utils.data.SubsetRandomSampler

```python
class torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

For more information, see [torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.SubsetRandomSampler).

## mindspore.dataset.SubsetRandomSampler

```python
class mindspore.dataset.SubsetRandomSampler(indices, num_samples=None)
```

For more information, see [mindspore.dataset.SubsetRandomSampler](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.SubsetRandomSampler.html).

## Differences

PyTorch: Samples the elements randomly from a sequence of indices, random generator can be set manually.

MindSpore: Samples the elements randomly from a sequence of indices, random generator is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter 1 | indices  | indices | - |
|     | Parameter 2 | generator  | -  | Specifies sampling logic. MindSpore uses global random sampling. |
|     | Parameter 3 | -   | num_samples  | Specify the number of samples returned by the sampler |

## Code Example

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
