# Differences with torch.utils.data.RandomSampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomSampler.md)

## torch.utils.data.RandomSampler

```python
class torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
```

For more information, see [torch.utils.data.RandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.RandomSampler).

## mindspore.dataset.RandomSampler

```python
class mindspore.dataset.RandomSampler(replacement=False, num_samples=None)
```

For more information, see [mindspore.dataset.RandomSampler](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.RandomSampler.html).

## Differences

PyTorch: Samples elements randomly, random generator can be set manually.

MindSpore: Samples elements randomly, random generator is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | data_source | -  | Dataset object to be sampled. MindSpore does not need this parameter. |
|     | Parameter2 | replacement   | replacement |- |
|     | Parameter3 | num_samples   | num_samples  |- |
|     | Parameter4 | generator  | -  | Specifies sampling logic. MindSpore uses global random sampling. |

### Code Example

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
