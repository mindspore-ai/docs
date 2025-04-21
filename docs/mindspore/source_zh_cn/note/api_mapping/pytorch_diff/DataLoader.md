# 比较与torch.utils.data.DataLoader的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/DataLoader.md)

## torch.utils.data.DataLoader

```python
class torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
    num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
    timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *,
    prefetch_factor=2, persistent_workers=False)
```

更多内容详见[torch.utils.data.DataLoader](https://pytorch.org/docs/1.8.1/data.html#torch.utils.data.DataLoader)。

## mindspore.dataset.GeneratorDataset

```python
class mindspore.dataset.GeneratorDataset(
    source, column_names=None, column_types=None, schema=None,
    num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None,
    num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=None)
```

更多内容详见[mindspore.dataset.GeneratorDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)。

## 差异对比

PyTorch：DataLoader需要接收数据集加载类、采样器，及批处理、混洗、多进程并行度等参数，以实现一个具有采样、分批、混洗等功能的数据迭代对象。其中`dataset`参数支持继承自`torch.utils.data.Dataset`的自定义类，或传入由`torchvision.datasets`、`torchtext.datasets`、`torchaudio.datasets`等组件中预定义好的数据集加载类。

MindSpore：GeneratorDataset需要接收数据集加载类、采样器，及混洗、分片、多进程并行性等参数，来创建一个用于数据迭代的迭代器。此API与PyTorch的DataLoader功能定位一样，均是用于加载自定义的数据集，但参数列表差异较大，下面的多个代码示例将演示如何使用2个API实现同样的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | dataset  | source    | - |
|     | 参数2 | batch_size   | - | MindSpore通过 `mindspore.dataset.Dataset.batch` 操作支持 |
|     | 参数3 | shuffle   | shuffle  | - |
|     | 参数4 | sampler  | sampler | - |
|     | 参数5 | batch_sampler   | - | MindSpore不支持 |
|     | 参数6 | num_workers  | num_parallel_workers   | - |
|     | 参数7 | collate_fn   | -  | MindSpore通过 `mindspore.dataset.Dataset.batch` 操作支持 |
|     | 参数8 | pin_memory   | -  | MindSpore不支持 |
|     | 参数9 | drop_last   | -  | MindSpore通过 `mindspore.dataset.Dataset.batch` 操作支持 |
|     | 参数10 | timeout   | -  | MindSpore不支持 |
|     | 参数11 | worker_init_fn   | -  | MindSpore不支持 |
|     | 参数12 | multiprocessing_context   | -  | 多进程上下文，MindSpore不支持 |
|     | 参数13 | generator   | -  | 自定义索引生成器，MindSpore不支持 |
|     | 参数14 | prefetch_factor   | -  | MindSpore通过 `mindspore.dataset.config.set_prefetch_size` 支持 |
|     | 参数15 | persistent_workers  | -  | 指定遍历完一次数据后是否释放数据集对象，MindSpore通过 `create_tuple_iterator` 的 `num_epoch` 参数支持，如果设置 `num_epoch` 大于1，则与 `persistent_workers` 为True一致 |
|     | 参数16 | -   | column_names   | 指定数据集生成的列名 |
|     | 参数17 | -   | column_types   | 指定生成数据集各个数据列的数据类型 |
|     | 参数18 | -   | schema   | 数据格式策略，用于指定读取数据列的数据类型、数据维度等信息 |
|     | 参数19 | -   | num_samples   | 指定从数据集中读取的样本数 |
|     | 参数20 | -   | num_shards   | 指定分布式训练时，将数据集进行划分的分片数 |
|     | 参数21 | -   | shard_id   | 指定分布式训练时，使用的分片ID号 |
|     | 参数22 | -   | python_multiprocessing    | 指定是否启用Python多进程模式加速运算 |
|     | 参数23 | -   | max_rowsize    | 指定在多进程之间复制数据时，共享内存分配的最大空间 |

### 代码示例1

> 定义一个迭代类型的数据集类与一个随机访问类型的数据集类，并通过DataLoader/GeneratorDataset加载。注意DataLoader的shuffle参数默认是False，GeneratorDataset的shuffle参数默认是True。

```python
# Torch
import torch

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0, shuffle=False)))
# Out: [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [1, 2, 3, 4]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
# Single-process loading
print(list(torch.utils.data.DataLoader(ds)))
# Out: [tensor([1]), tensor([2]), tensor([3]), tensor([4])]
```

```python
# MindSpore
import mindspore as ms

class MyIterableDataset():
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
# Single-process loading
print(list(ms.dataset.GeneratorDataset(ds, column_names=["data"], num_parallel_workers=1, shuffle=False)))
# Out: [[Tensor(shape=[], dtype=Int64, value= 3)], [Tensor(shape=[], dtype=Int64, value= 4)], [Tensor(shape=[], dtype=Int64, value= 5)], [Tensor(shape=[], dtype=Int64, value= 6)]]

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [1, 2, 3, 4]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
# Single-process loading
print(list(ms.dataset.GeneratorDataset(ds, column_names=["data"], shuffle=False)))
# Out: [[Tensor(shape=[], dtype=Int64, value= 1)], [Tensor(shape=[], dtype=Int64, value= 2)], [Tensor(shape=[], dtype=Int64, value= 3)], [Tensor(shape=[], dtype=Int64, value= 4)]]
```

### 代码示例2

> 定义一个数据集类，并对数据进行batch为2的批处理。

```python
# Torch
import torch

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [1, 2, 3, 4, 5]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
dataloader = torch.utils.data.DataLoader(ds, batch_size=2, drop_last=True)
print(list(dataloader))
# Out: [tensor([1, 2]), tensor([3, 4])]
```

```python
# MindSpore
import mindspore as ms

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [1, 2, 3, 4, 5]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

ds = MyMapDataset()
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], shuffle=False)
dataloader = dataloader.batch(2, drop_remainder=True)
print(list(dataloader))
# Out: [[Tensor(shape=[2], dtype=Int64, value= [1, 2])], [Tensor(shape=[2], dtype=Int64, value= [3, 4])]]
```

### 代码示例3

> 定义一个数据集类，进行批处理时引入collate_fn逻辑。

```python
# Torch
import torch

class MyMapDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = torch.Tensor([1, 2, 3, 4, 5])
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def my_collate_fn(batch):
    for i, _ in enumerate(batch):
        batch[i] = batch[i] + 2
    return torch.stack(batch)

ds = MyMapDataset()
dataloader = torch.utils.data.DataLoader(ds, batch_size=2, drop_last=True, collate_fn=my_collate_fn)
print(list(dataloader))
# Out: [tensor([3., 4.]), tensor([5., 6.])]
```

```python
# MindSpore
import mindspore as ms
import numpy as np

class MyMapDataset():
    def __init__(self):
        super(MyMapDataset).__init__()
        self.data = [1, 2, 3, 4, 5]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def my_collate_fn(batch, batchinfo):
    for i, _ in enumerate(batch):
        batch[i] = batch[i] + 2
    return np.stack(batch),

ds = MyMapDataset()
dataloader = ms.dataset.GeneratorDataset(ds, column_names=["data"], shuffle=False)
dataloader = dataloader.batch(2, drop_remainder=True, per_batch_map=my_collate_fn)
print(list(dataloader))
# Out: [[Tensor(shape=[2], dtype=Int64, value= [3, 4])], [Tensor(shape=[2], dtype=Int64, value= [5, 6])]]
```
