# Constructing Dataset

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/migration_guide/model_development/dataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

This chapter focuses on considerations related to data processing in network migration. For basic data processing, please refer to:

[Data Processing](https://www.mindspore.cn/tutorials/en/r1.9/beginner/dataset.html)

[Auto Augmentation](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/augment.html)

[Lightweight Data Processing](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/eager.html)

[Optimizing the Data Processing](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/optimize.html)

## Basic Process of Data Construction

The whole basic process of data construction consists of two main aspects: dataset loading and data augmentation.

### Loading Dataset

MindSpore provides interfaces for loading many common datasets. The most used ones are as follows:

| Data Interfaces | Introduction |
| -------| ---- |
| [Cifar10Dataset](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset) | Cifar10 dataset read interface (you need to download the original bin file of Cifar10) |
| [MNIST](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset) | Minist handwritten digit recognition dataset (you need to download the original file) |
| [ImageFolderDataset](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.ImageFolderDataset.html) | Dataset reading method of using file directories as the data organization format for classification (common for ImageNet) |
| [MindDataset](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset) | MindRecord data read interface |
| [GeneratorDataset](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.GeneratorDataset.html) | customized data interface |
| [FakeImageDataset](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/mindspore.dataset.FakeImageDataset.html) | Constructing a fake image dataset |

For common dataset interfaces in different fields, refer to [Loading interfaces to common datasets](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.html).

#### GeneratorDataset, A Custom Dataset

The basic process for constructing a custom Dataset object is as follows:

First create an iterator class and define three methods `__init__`, `__getitem__` and `__len__` in that class.

```python
import numpy as np
class MyDataset():
    """Self Defined dataset."""
    def __init__(self, n):
        self.data = []
        self.label = []
        for _ in range(n):
            self.data.append(np.zeros((3, 4, 5)))
            self.label.append(np.ones((1)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label
```

> It is better not to use MindSpore operators in the iterator class, because it usually adds multiple threads in the data processing phase, which may cause problems.
>
> The output of the iterator needs to be a numpy array.
>
> The iterator must set the `__len__` method, and the returned result must be the real dataset size. Setting it larger will cause problems when gettingitem takes values.

Then use GeneratorDataset to encapsulate the iterator class:

```python
import mindspore.dataset as ds

my_dataset = MyDataset(10)
# corresponding to torch.utils.data.DataLoader(my_dataset)
dataset = ds.GeneratorDataset(my_dataset, column_names=["data", "label"])
```

When customizing the dataset, you need to set a name for each output, such as `column_names=["data", "label"]` above, indicating that the first output column of the iterator is `data` and the second is `label`. In the subsequent data augmentation and data iteration obtaining phases, the different columns can be processed separately by name.

**All MindSpore data interfaces** have some general attribute controls, and some of the common ones are described here:

| Attributes | Introduction |
| ---- | ---- |
| num_samples(int) | Specify the total number of data samples |
| shuffle(bool)  | Whether to do random disruptions to the data |
| sampler(Sampler) | Data sampler, customizing data disruption, allocation. `sampler` setting and `num_shards`, `shard_id` mutually exclusive |
| num_shards(int) | Used in distributed scenarios to divide data into several parts, used in conjunction with `shard_id` |
| shard_id(int) | For distributed scenarios, taking nth data (n ranges from 0 to n-1, and n is the set `num_shards`), used in conjunction with `num_shards` |
| num_parallel_workers(int) | Number of threads in parallel configuration |

For example:

```python
import mindspore.dataset as ds

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32, 32, 3), num_classes=10, base_seed=0)
print(dataset.get_dataset_size())
# 1000

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32, 32, 3), num_classes=10, base_seed=0, num_samples=3)
print(dataset.get_dataset_size())
# 3

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32, 32, 3), num_classes=10, base_seed=0,
                              num_shards=8, shard_id=0)
print(dataset.get_dataset_size())
# 1000 / 8 = 125
```

```text
1000
3
125
```

### Data Processing and Augmentation

MindSpore dataset object uses the map interface for data augmentation. See [map Interface](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.map.html#mindspore.dataset.Dataset.map)

```text
map(operations, input_columns=None, output_columns=None, column_order=None, num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16, offload=None)
```

Given a set of data augmentation lists, data augmentations are applied to the dataset objects in order.

Each data augmentation operation takes one or more data columns in the dataset object as input and outputs the result of the data augmentation as one or more data columns. The first data augmentation operation takes the specified columns in input_columns as input. If there are multiple data augmentation operations in the data augmentation list, the output columns of the previous data augmentation will be used as input columns for the next data augmentation.

The column name of output column in the last data augmentation is specified by `output_columns`. If `output_columns` is not specified, the output column name is the same as `input_columns`.

The above introduction may be tedious, but in short, `map` is to do the operations specified in `operations` on some columns of the dataset. Here `operations` can be the data augmentation provided by MindSpore.

[audio](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.audio.html), [text](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.text.html), [vision](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.vision.html), and [transforms](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.transforms.html). For more details, refer to [Data Transforms](https://www.mindspore.cn/tutorials/en/r1.9/beginner/transforms.html), which is also a method of python. You can use opencv, PIL, pandas and some other third party methods, like loading dataset. **Don't use MindSpore operators**.

MindSpore also provides some common random augmentation methods: [Auto augmentation](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/augment.html). When using data augmentation specifically, it is best to read [Optimizing the Data Processing](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/optimize.html) in the recommended order.

At the end of data augmentation, you can use the batch operator to merge `batch_size` pieces of consecutive data in the dataset into a single batch data. For details, please refer to [batch](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html#mindspore.dataset.dataset.batch). Note that the parameter `drop_remainder` needs to be set to True during training and False during inference.

```python
import mindspore.dataset as ds

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32, 32, 3), num_classes=10, base_seed=0)\
    .batch(32, drop_remainder=True)
print(dataset.get_dataset_size())
# 1000 // 32 = 31

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32, 32, 3), num_classes=10, base_seed=0)\
    .batch(32, drop_remainder=False)
print(dataset.get_dataset_size())
# ceil(1000 / 32) = 32
```

```text
31
32
```

The batch operator can also use some augmentation operations within batch. For details, see [YOLOv3](https://gitee.com/mindspore/models/blob/r1.9/official/cv/yolov3_darknet53/src/yolo_dataset.py#L177).

## Data Iteration

MindSpore data objects are obtained iteratively in the following ways.

### [create_dict_iterator](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_dict_iterator.html#mindspore.dataset.Dataset.create_dict_iterator)

Creates an iterator based on the dataset object, and the output data is of dictionary type.

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)
iterator = dataset.create_dict_iterator()
for data_dict in iterator:
    for name in data_dict.keys():
        print(name, data_dict[name].shape)
    print("="*20)
```

```text
image (10, 32, 32, 3)
label (10,)
====================
image (10, 32, 32, 3)
label (10,)
====================
```

### [create_tuple_iterator](https://www.mindspore.cn/docs/en/r1.9/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_tuple_iterator.html#mindspore.dataset.Dataset.create_tuple_iterator)

Create an iterator based on the dataset object, and output data is a list of `numpy.ndarray` data.

You can specify all column names and the order of the columns in the output by the parameter `columns`. If columns is not specified, the order of the columns will remain the same.

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)
iterator = dataset.create_tuple_iterator()
for data_tuple in iterator:
    for data in data_tuple:
        print(data.shape)
    print("="*20)
```

```text
(10, 32, 32, 3)
(10,)
====================
(10, 32, 32, 3)
(10,)
====================
```

### Traversing Directly over dataset Objects

> Note that this writing method does not `shuffle` after traversing an epoch, so it may affect the precision when used in training. The above two methods are recommended when direct data iterations are needed during training.

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)

for data in dataset:
    for data in data:
        print(data.shape)
    print("="*20)
```

```text
(10, 32, 32, 3)
(10,)
====================
(10, 32, 32, 3)
(10,)
====================
```

The latter two of these can be used directly when the order of data read is the same as the order required by the network.

```python
for data in dataset:
    loss = net(*data)
```
