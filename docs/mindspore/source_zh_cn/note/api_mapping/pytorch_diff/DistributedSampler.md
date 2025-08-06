# 比较与torch.utils.data.distributed.DistributedSampler的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/DistributedSampler.md)

## torch.utils.data.distributed.DistributedSampler

```python
class torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)
```

更多内容详见[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.distributed.DistributedSampler)。

## mindspore.dataset.DistributedSampler

```python
class mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=-1)
```

更多内容详见[mindspore.dataset.DistributedSampler](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.DistributedSampler.html)。

## 差异对比

PyTorch：将数据集进行分片，可用于分布式训练。`drop_last` 参数控制是否丢弃多余的数据或对每个设备上的数据进行补齐， `seed` 参数控制混洗的随机种子。

MindSpore：将数据集进行分片，可用于分布式训练。不支持自动丢弃多余数据或补齐每个设备上的数据，不支持在混洗时指定随机种子。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | dataset | -  | 被采样的数据集对象，MindSpore不需要传入 |
|     | 参数2 | num_replicas  | num_shards |- |
|     | 参数3 | rank  | shard_id  |- |
|     | 参数4 | shuffle  | shuffle  |- |
|     | 参数5 | seed | - | shuffle参数为True时的采样种子，MindSpore不支持 |
|     | 参数6 | drop_last  | - | 控制是否丢弃平均分配后多余的数据，或补全数据使得分片后多卡的数据一致，MindSpore不支持 |
|     | 参数7 | -  | num_samples  | 用于部分获取采样得到的样本 |
|     | 参数8 | -  | offset  | 分布式采样结果进行分配时的起始分片ID号，从不同的分片ID开始分配数据，可能会影响每个分片的最终样本数 |

## 代码示例

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
