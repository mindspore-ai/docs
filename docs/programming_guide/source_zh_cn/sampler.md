# 采样器

<!-- TOC -->

- [采样器](#采样器)
    - [概述](#概述)
    - [MindSpore采样器](#mindspore采样器)
        - [SequentialSampler](#sequentialsampler)
        - [RandomSampler](#randomsampler)
        - [WeightedRandomSampler](#weightedrandomsampler)
        - [SubsetRandomSampler](#subsetrandomsampler)
        - [PKSampler](#pksampler)
        - [DistributedSampler](#distributedsampler)
    - [自定义采样器](#自定义采样器)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/sampler.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore提供了多种用途的采样器，帮助用户对数据集进行不同形式的采样，以满足训练需求，能够解决诸如数据集过大或样本类别分布不均等问题。只需在加载数据集时将采样器对象传入，即可实现数据的采样。

MindSpore目前提供的采样器类别如下表所示。此外，用户也可以根据需要实现自定义的采样器类。

| 采样器名称  | 采样器说明 |
| ----  | ----           |
| SequentialSampler | 顺序采样器，按照数据的原始顺序采样指定数目的数据。 |
| RandomSampler | 随机采样器，在数据集中随机地采样指定数目的数据。 |
| WeightedRandomSampler | 带权随机采样器，在每种类别的数据中按照指定概率随机采样指定数目的数据。 |
| SubsetRandomSampler | 子集随机采样器，在指定的索引范围内随机采样指定数目的数据。 |
| PKSampler | PK采样器，在指定的数据集类别P中，每种类别各采样K条数据。 |
| DistributedSampler | 分布式采样器，在分布式训练中对数据集分片进行采样。 |

## MindSpore采样器

下面以[CIFAR10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)为例，介绍MindSpore采样器使用方法。

### SequentialSampler

从指定的索引位置开始顺序采样指定数目的数据。

```python
# 通过SequentialSampler定义一个顺序采样器，并作用于数据集

import mindspore.dataset as ds

# CIFAR10数据集路径
DATA_DIR = "Cifar10Data/"

# 1. 定义一个顺序采样器SequentialSampler，按照读取顺序获取5个样本数据
sampler = ds.SequentialSampler(num_samples=5)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出5个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])

print("")

# 2. 定义一个顺序采样器SequentialSampler，跳过前2个数据，继续按照读取顺序获取5个样本数据
sampler = ds.SequentialSampler(start_index=2, num_samples=5)
dataset2 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出5个样本数据
for data in dataset2.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])

print("")

# 3. 同类用法，指定数据集中的num_samples参数为5，shuffle参数为False，同样可以达到1的效果
dataset3 = ds.Cifar10Dataset(DATA_DIR, num_samples=5, shuffle=False)

# 启动数据管道，输出5个样本数据
for data in dataset3.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 4

Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 5
Image shape: (32, 32, 3) , Label: 6

Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 4
```

### RandomSampler

从索引序列中随机采样指定数目的数据。

```python
# 通过RandomSampler定义一个随机采样器，并作用于数据集

import mindspore.dataset as ds

# 设置全局随机种子，确保RandomSampler的行为可预测
ds.config.set_seed(0)

# CIFAR数据集路径
DATA_DIR = "Cifar10Data/"

# 1. 定义一个随机采样器SequentialSampler，随机获取5个样本数据
sampler = ds.RandomSampler(num_samples=5)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出5个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])

print("")

# 2. 定义一个随机采样器RandomSampler，replacement=True意味着有放回抽样
sampler = ds.RandomSampler(replacement=True, num_samples=5)
dataset2 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出5个样本数据
for data in dataset2.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])

print("")

# 3. 同类用法，指定数据集中的num_samples参数为5，shuffle参数为True，同样可以达到2的效果
dataset3 = ds.Cifar10Dataset(DATA_DIR, num_samples=5, shuffle=True)

# 启动数据管道，输出5个样本数据
for data in dataset3.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 6

Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 7

Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 8
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 7
```

### WeightedRandomSampler

指定每种类别的采样概率，按照概率在各类别中随机采样指定数目的数据。

```python
# 通过WeightedRandomSampler定义一个带权重的随机采样器，并作用于数据集

import mindspore.dataset as ds

# 设置全局随机种子，确保WeightedRandomSampler的行为可预测
ds.config.set_seed(1)

# CIFAR数据集路径
DATA_DIR = "Cifar10Data/"

# 定义一个带权重的随机采样器WeightedRandomSampler
# weights代表CIFAR10中10类数据的采样概率，num_samples表示随机获取6个样本数据
# replacement参数与RandomSampler中一致
weights = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
sampler = ds.WeightedRandomSampler(weights, num_samples=6)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出6个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 0
```

### SubsetRandomSampler

从指定索引子序列中随机采样指定数目的数据。

```python
# 通过SubsetRandomSampler定义一个子集随机采样器，并作用于数据集

import mindspore.dataset as ds

# 设置全局随机种子，确保SubsetRandomSampler的行为可预测
ds.config.set_seed(2)

# CIFAR数据集路径
DATA_DIR = "Cifar10Data/"

# 定义一个带采样集合的随机采样器SubsetRandomSampler
# indice代表可采样的集合，num_samples表示获取3个样本数据，即从可采样集合中(0~5)随机获取3个值，作为下标访问数据集的数据
indices = [0, 1, 2, 3, 4, 5]
sampler = ds.SubsetRandomSampler(indices, num_samples=3)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出3个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 5
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 3
```

### PKSampler

在指定的数据集类别P中，每种类别各采样K条数据。

```python
# 通过PKSampler定义一个针对各个类别随机采样器，并作用于数据集

import mindspore.dataset as ds

# 设置全局随机种子，确保PKSampler的shuffle参数行为可预测
ds.config.set_seed(3)

# CIFAR数据集路径
DATA_DIR = "Cifar10Data/"

# 定义一个针对类别采样的随机采样器PKSampler
# num_val代表从每个类别采样K个样本，class_column代表针对特定的数据列采样（一般是label）
# num_samples代表输出的样本数，设置num_samples = num_val*class_nums，确保每个类别平均采样
# shuffle代表样本是否需要被混洗
sampler = ds.PKSampler(num_val=2, class_column='label', num_samples=20)
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出20个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
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

```python
# 通过DistributedSampler定义一个将数据集进行分片操作，并获取某个分片进行采样的采样器，并作用于数据集

import numpy as np
import mindspore.dataset as ds

# 构建一个list
data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 定义一个采样器DistributedSampler
# num_shards代表将CIFAR数据集拆分成n个分片
# shard_id代表获取第m个分片
# num_samples代表获取该分片的10个样本
# shuffle代表样本是否需要被混洗
sampler = ds.DistributedSampler(num_shards=3, shard_id=0, shuffle=False, num_samples=3)

# 从list中构建数据管道
dataset = ds.NumpySlicesDataset(data_source, column_names=["data"], sampler=sampler)

# 经过DistributedSampler分片后，数据集的内容为
# shard_id 0: 0, 3, 6
# shard_id 1: 1, 4, 7
# shard_id 2: 2, 5, 8
# 因此第0个分片拥有数据为0, 3, 6
for data in dataset.create_dict_iterator():
    print(data)
```

```
{'data': Tensor(shape=[], dtype=Int64, value= 0)}
{'data': Tensor(shape=[], dtype=Int64, value= 3)}
{'data': Tensor(shape=[], dtype=Int64, value= 6)}
```

## 自定义采样器

用户可以继承Sampler基类，通过实现`__iter__`方法来自定义采样器的采样方式。

```python
# 继承Sampler基类，重载__iter__成为新的采样器

import mindspore.dataset as ds

class MySampler(ds.Sampler):
    def __iter__(self):
        # 采样器的行为是，从下标0开始到下标9，以2为间隔采样
        for i in range(0, 10, 2):
            yield i

# CIFAR数据集路径
DATA_DIR = "Cifar10Data/"

# 将自定义构建的采样器传入到sampler参数
dataset1 = ds.Cifar10Dataset(DATA_DIR, sampler=MySampler())

# 启动数据管道，输出5个样本数据
for data in dataset1.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 8
```
