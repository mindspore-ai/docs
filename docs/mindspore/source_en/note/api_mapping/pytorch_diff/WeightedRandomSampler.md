# Differences with torch.utils.data.WeightedRandomSampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/WeightedRandomSampler.md)

## torch.utils.data.WeightedRandomSampler

```python
class torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
```

For more information, see [torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.WeightedRandomSampler).

## mindspore.dataset.WeightedRandomSampler

```python
class mindspore.dataset.WeightedRandomSampler(weights, num_samples=None, replacement=True)
```

For more information, see [mindspore.dataset.WeightedRandomSampler](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.WeightedRandomSampler.html).

## Differences

PyTorch: Given a weight list for a sample, the sample is sampled according to the magnitude of the weights. Specifying sampling logic is supported.

MindSpore: Given a weight list for a sample, the sample is sampled according to the magnitude of the weights. Specifying sampling logic is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | weights  | weights   | -|
|     | Parameter2 | num_samples    | num_samples  |- |
|     | Parameter3 | replacement    | replacement   |- |
|     | Parameter4 | generator  | -  | Specifies sampling logic. MindSpore uses global random sampling. |

## Code Example

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
