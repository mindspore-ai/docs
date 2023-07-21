# 采样器

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/sampler.md)
&nbsp;&nbsp;
[![查看Notebook](./_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_sampler.ipynb)
&nbsp;&nbsp;
[![在线运行](./_static/logo_modelarts.png)](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV9zYW1wbGVyLmlweW5i&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c)

## 概述

MindSpore提供了多种用途的采样器（Sampler），帮助用户对数据集进行不同形式的采样，以满足训练需求，能够解决诸如数据集过大或样本类别分布不均等问题。只需在加载数据集时传入采样器对象，即可实现数据的采样。

MindSpore目前提供的部分采样器类别如下表所示。此外，用户也可以根据需要实现自定义的采样器类。更多采样器的使用方法参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.dataset.html)。

| 采样器名称  | 采样器说明 |
| ----  | ----           |
| RandomSampler | 随机采样器，在数据集中随机地采样指定数目的数据。 |
| WeightedRandomSampler | 带权随机采样器，依照长度为N的概率列表，在前N个样本中随机采样指定数目的数据。 |
| SubsetRandomSampler | 子集随机采样器，在指定的索引范围内随机采样指定数目的数据。 |
| PKSampler | PK采样器，在指定的数据集类别P中，每种类别各采样K条数据。 |
| DistributedSampler | 分布式采样器，在分布式训练中对数据集分片进行采样。 |

## MindSpore采样器

下面以CIFAR-10数据集为例，介绍几种常用MindSpore采样器的使用方法。下载[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)并解压，目录结构如下。

```text
└─cifar-10-batches-bin
    ├── batches.meta.txt
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── readme.html
    └── test_batch.bin
```

### RandomSampler

从索引序列中随机采样指定数目的数据。

下面的样例使用随机采样器分别从CIFAR-10数据集中有放回和无放回地随机采样5个数据，并展示已加载数据的形状和标签。

```python
import mindspore.dataset as ds

ds.config.set_seed(0)

DATA_DIR = "cifar-10-batches-bin/"

print("------ Without Replacement ------")

sampler = ds.RandomSampler(num_samples=5)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])

print("------ With Replacement ------")

sampler = ds.RandomSampler(replacement=True, num_samples=5)
dataset2 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset2.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
------ Without Replacement ------
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 7
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 4
------ With Replacement ------
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 5
```

### WeightedRandomSampler

指定长度为N的采样概率列表，按照概率在前N个样本中随机采样指定数目的数据。

下面的样例使用带权随机采样器从CIFAR-10数据集的前10个样本中按概率获取6个样本，并展示已读取数据的形状和标签。

```python
import mindspore.dataset as ds

ds.config.set_seed(1)

DATA_DIR = "cifar-10-batches-bin/"

weights = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
sampler = ds.WeightedRandomSampler(weights, num_samples=6)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 6
```

### SubsetRandomSampler

从指定索引子序列中随机采样指定数目的数据。

下面的样例使用子序列随机采样器从CIFAR-10数据集的指定子序列中抽样3个样本，并展示已读取数据的形状和标签。

```python
import mindspore.dataset as ds

ds.config.set_seed(2)

DATA_DIR = "cifar-10-batches-bin/"

indices = [0, 1, 2, 3, 4, 5]
sampler = ds.SubsetRandomSampler(indices, num_samples=3)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 4
```

### PKSampler

在指定的数据集类别P中，每种类别各采样K条数据。

下面的样例使用PK采样器从CIFAR-10数据集中每种类别抽样2个样本，最多20个样本，并展示已读取数据的形状和标签。

```python
import mindspore.dataset as ds

ds.config.set_seed(3)

DATA_DIR = "cifar-10-batches-bin/"

sampler = ds.PKSampler(num_val=2, class_column='label', num_samples=20)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 5
Image shape: (32, 32, 3) , Label: 5
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 7
Image shape: (32, 32, 3) , Label: 7
Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 9
```

### DistributedSampler

在分布式训练中，对数据集分片进行采样。

下面的样例使用分布式采样器将构建的数据集分为3片，在每个分片中采样不多于3个数据样本，并展示第0个分片读取到的数据。

```python
import numpy as np
import mindspore.dataset as ds

data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]

sampler = ds.DistributedSampler(num_shards=3, shard_id=0, shuffle=False, num_samples=3)
dataset = ds.NumpySlicesDataset(data_source, column_names=["data"], sampler=sampler)

for data in dataset.create_dict_iterator():
    print(data)
```

输出结果如下：

```text
{'data': Tensor(shape=[], dtype=Int64, value= 0)}
{'data': Tensor(shape=[], dtype=Int64, value= 3)}
{'data': Tensor(shape=[], dtype=Int64, value= 6)}
```

## 自定义采样器

用户可以继承`Sampler`基类，通过实现`__iter__`方法来自定义采样器的采样方式。

下面的样例定义了一个从下标0至下标9间隔为2采样的采样器，将其作用于CIFAR-10数据集，并展示已读取数据的形状和标签。

```python
import mindspore.dataset as ds

class MySampler(ds.Sampler):
    def __iter__(self):
        for i in range(0, 10, 2):
            yield i

DATA_DIR = "cifar-10-batches-bin/"

dataset = ds.Cifar10Dataset(DATA_DIR, sampler=MySampler())

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 8
```
