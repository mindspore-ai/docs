# Data Sampling

`Ascend` `GPU` `CPU` `Data Preparation`

<!-- TOC -->

- [Data Sampling](#data-sampling)
    - [Overview](#overview)
    - [MindSpore Samplers](#mindspore-samplers)
        - [RandomSampler](#randomsampler)
        - [WeightedRandomSampler](#weightedrandomsampler)
        - [SubsetRandomSampler](#subsetrandomsampler)
        - [PKSampler](#pksampler)
        - [DistributedSampler](#distributedsampler)
    - [User-defined Sampler](#user-defined-sampler)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/sampler.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore provides multiple samplers to help you sample datasets for various purposes to meet training requirements and solve problems such as oversized datasets and uneven distribution of sample categories. You only need to import the sampler object when loading the dataset for sampling the data.

The following table lists part of the common samplers supported by MindSpore. In addition, you can define your own sampler class as required. For more samplers, see [MindSpore API](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.html).

| Sampler | Description |
| ----  | ----           |
| RandomSampler | Random sampler, which randomly samples a specified amount of data from a dataset.  |
| WeightedRandomSampler | Weighted random sampler, which randomly samples a specified amount of data from the first N samples based on the specified probability list with the length of N.  |
| SubsetRandomSampler | Subset random sampler, which randomly samples a specified amount of data within a specified index range.  |
| PKSampler | PK sampler, which samples K pieces of data from the specified P categories.  |
| DistributedSampler | Distributed sampler, which samples dataset shards in distributed training.  |

## MindSpore Samplers

The following uses the CIFAR-10 as an example to introduce several common MindSpore samplers.

Download the CIFAR-10 data set and unzip it to the specified path, execute the following command:

```bash
wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz --no-check-certificate
mkdir -p datasets
tar -xzf cifar-10-binary.tar.gz -C datasets
mkdir -p datasets/cifar-10-batches-bin/train datasets/cifar-10-batches-bin/test
mv -f datasets/cifar-10-batches-bin/test_batch.bin datasets/cifar-10-batches-bin/test
mv -f datasets/cifar-10-batches-bin/data_batch*.bin datasets/cifar-10-batches-bin/batches.meta.txt datasets/cifar-10-batches-bin/train
tree ./datasets/cifar-10-batches-bin
```

```text
./datasets/cifar-10-batches-bin
├── readme.html
├── test
│   └── test_batch.bin
└── train
    ├── batches.meta.txt
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    └── data_batch_5.bin

2 directories, 8 files
```

### RandomSampler

Randomly samples a specified amount of data from the index sequence.

The following example uses a random sampler to randomly sample five pieces of data from the CIFAR-10 dataset with and without replacement, and displays shapes and labels of the loaded data.

```python
import mindspore.dataset as ds

ds.config.set_seed(0)

DATA_DIR = "./datasets/cifar-10-batches-bin/train/"

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

The output is as follows:

```text
------ Without Replacement ------
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 4
------ With Replacement ------
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
```

### WeightedRandomSampler

Specifies a sampling probability list with the length of N and randomly samples a specified amount of data from the first N samples based on the probability.

The following example uses a weighted random sampler to obtain 6 samples based on probability from the first 10 samples in the CIFAR-10 dataset, and displays shapes and labels of the loaded data.

```python
import mindspore.dataset as ds

ds.config.set_seed(1)

DATA_DIR = "./datasets/cifar-10-batches-bin/train/"

weights = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
sampler = ds.WeightedRandomSampler(weights, num_samples=6)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

The output is as follows:

```text
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 6
```

### SubsetRandomSampler

Randomly samples a specified amount of data from the specified index subset.

The following example uses a subset random sampler to obtain 3 samples from the specified subset in the CIFAR-10 dataset, and displays shapes and labels of the loaded data.

```python
import mindspore.dataset as ds

ds.config.set_seed(2)

DATA_DIR = "./datasets/cifar-10-batches-bin/train/"

indices = [0, 1, 2, 3, 4, 5]
sampler = ds.SubsetRandomSampler(indices, num_samples=3)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

The output is as follows:

```text
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
```

### PKSampler

Samples K pieces of data from the specified P categories.

The following example uses the PK sampler to obtain 2 samples from each category in the CIFAR-10 dataset, not more than 20 samples in total, and displays shapes and labels of the read data.

```python
import mindspore.dataset as ds

ds.config.set_seed(3)

DATA_DIR = "./datasets/cifar-10-batches-bin/train/"

sampler = ds.PKSampler(num_val=2, class_column='label', num_samples=20)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

The output is as follows:

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

Samples dataset shards in distributed training.

The following example uses a distributed sampler to divide a generated dataset into three shards, obtains no more than three data samples in each shard, and displays the loaded data on shard number 0.

```python
import mindspore.dataset as ds

data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]

sampler = ds.DistributedSampler(num_shards=3, shard_id=0, shuffle=False, num_samples=3)
dataset = ds.NumpySlicesDataset(data_source, column_names=["data"], sampler=sampler)

for data in dataset.create_dict_iterator():
    print(data)
```

The output is as follows:

```text
{'data': Tensor(shape=[], dtype=Int64, value= 0)}
{'data': Tensor(shape=[], dtype=Int64, value= 3)}
{'data': Tensor(shape=[], dtype=Int64, value= 6)}
```

## User-defined Sampler

You can inherit the `Sampler` base class and define the sampling method of the sampler by implementing the `__iter__` method.

The following example defines a sampler with an interval of 2 samples from subscript 0 to subscript 9, applies the sampler to the CIFAR-10 dataset, and displays shapes and labels of the read data.

```python
import mindspore.dataset as ds

class MySampler(ds.Sampler):
    def __iter__(self):
        for i in range(0, 10, 2):
            yield i

DATA_DIR = "./datasets/cifar-10-batches-bin/train/"

dataset = ds.Cifar10Dataset(DATA_DIR, sampler=MySampler())

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

The output is as follows:

```text
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 8
```
