# Differences with torch.utils.data.SequentialSampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SequentialSampler.md)

## torch.utils.data.SequentialSampler

```python
class torch.utils.data.SequentialSampler(data_source)
```

For more information, see [torch.utils.data.SequentialSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.SequentialSampler).

## mindspore.dataset.SequentialSampler

```python
class mindspore.dataset.SequentialSampler(start_index=None, num_samples=None)
```

For more information, see [mindspore.dataset.SequentialSampler](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.SequentialSampler.html).

## Differences

PyTorch: Samples elements sequentially.

MindSpore: Samples elements sequentially. Support for specifying sequential indexing and sample size.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | data_source | -  | Dataset object to be sampled. MindSpore does not need this parameter. |
|     | Parameter2 | -   | start_index  | Index to start sampling at |
|     | Parameter3 | -   | num_samples  | Specify the number of samples returned by the sampler |

## Code Example

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
