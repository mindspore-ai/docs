# Differences with torch.utils.data.distributed.DistributedSampler

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/DistributedSampler.md)

## torch.utils.data.distributed.DistributedSampler

```python
class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)
```

For more information, see [torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.distributed.DistributedSampler).

## mindspore.dataset.DistributedSampler

```python
class mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=-1)
```

For more information, see [mindspore.dataset.DistributedSampler](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.DistributedSampler.html).

## Differences

PyTorch: Split datasets into shards for distributed training. The `drop_last` parameter can be used to control whether to drop the tail of the data to make it evenly divisible across the devices, `seed` controls the random seed of shuffle.

MindSpore: Split datasets into shards for distributed training. Drop data or make it evenly divisible across the devices is not supported, random seed is not supported to specified for shuffle.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter  | Parameter1 | dataset | -  | Dataset object to be sampled. MindSpore does not need this parameter. |
|     | Parameter2 | num_replicas  | num_shards |- |
|     | Parameter3 | rank  | shard_id  |- |
|     | Parameter4 | shuffle  | shuffle  |- |
|     | Parameter5 | seed | - | Sampling seed when `shuffle` is set to True. MindSpore doesn't support this parameter. |
|     | Parameter6 | drop_last  | - | Controls whether to drop the tail of the data to make it evenly divisible across the devices. MindSpore doesn't support this parameter. |
|     | Parameter7 | -  | num_samples  |  Used to obtain partial samples. |
|     | Parameter8 | -  | offset  | The starting shard ID where the elements in the dataset are sent to. |

## Code Example

```python
import torch
from torch.utils.data.distributed import DistributedSampler

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False)
dataloader = torch.utils.data.DataLoader(ds, sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# tensor([1])
# tensor([3])
```

```python
import mindspore as ms
from mindspore.dataset import DistributedSampler

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [i for i in range(4)]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
sampler = DistributedSampler(num_shards=2, shard_id=1, shuffle=False)
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], sampler=sampler)

for data in dataloader:
    print(data)
# Out:
# [Tensor(shape=[], dtype=Int64, value= 1)]
# [Tensor(shape=[], dtype=Int64, value= 3)]
```
