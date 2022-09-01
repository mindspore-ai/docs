# 数据集构建

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/model_development/dataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本章节主要对网络迁移中数据处理相关的注意事项加以说明，基本的数据处理请参考：

[数据处理](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html)

[自动数据增强](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/augment.html)

[轻量化数据处理](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/eager.html)

[数据处理性能优化](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/optimize.html)

## 数据构建基本流程

整个数据构建基本流程主要包括：数据集加载和数据增强这两方面。

### 数据集加载

MindSpore提供了很多常见数据集的加载接口, 用的比较多的接口有：

| 数据接口 | 介绍 |
| -------| ---- |
| [Cifar10Dataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset) | Cifar10 数据集读入接口（需要自行下载Cifar10的原始bin文件） |
| [MNIST](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset) | Minist 手写数字识别数据集（需要自行下载原始文件） |
| [ImageFolderDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.ImageFolderDataset.html) | 以文件目录作为分类的数据组织格式的数据集读取方式（ImageNet常用） |
| [MindDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset) | MindRecord数据读入接口 |
| [GeneratorDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html) | 用户自定义数据接口 |
| [FakeImageDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.FakeImageDataset.html) | 构造一个假的图像数据集 |

还有很多常用的不同领域的数据集接口，详情参考[常见数据集的加载接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html)。

#### 自定义数据集 GeneratorDataset

构造自定义 Dataset 对象的基本流程如下：

首先创建一个迭代器类，在该类中定义 __init__ 、 __getitem__ 、 __len__ 三个方法：

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

> 在迭代器类中最好不要使用MindSpore的算子，因为一般在数据处理阶段会加多线程，这样会有问题。
>
> 迭代器的输出需要是numpy的array。
>
> 迭代器必须要设置__len__方法，返回的结果一定要是真实的数据集大小，设置大了在getitem取值时会有问题。

然后使用 GeneratorDataset 封装迭代器类：

```python
import mindspore.dataset as ds

my_dataset = MyDataset(10)
# corresponding to torch.utils.data.DataLoader(my_dataset)
dataset = ds.GeneratorDataset(my_dataset, column_names=["data", "label"])
```

在自定义数据集时需要给每一个输出设置一个名字，如上面的`column_names=["data", "label"]`，表示迭代器的第一个输出列叫`data`，第二个叫`label`。在后续的数据增强以及数据迭代获取阶段，可以通过名字来分别对不同列进行处理。

**所有MindSpore的数据接口**，有一些通用的属性控制，这里介绍一些常用的：

| 属性 | 介绍 |
| ---- | ---- |
| num_samples(int) | 规定数据总的sample数 |
| shuffle(bool)  | 是否对数据做随机打乱 |
| sampler(Sampler) | 数据取样器，可以自定义数据打乱、分配，`sampler`设置和`num_shards`、`shard_id`互斥 |
| num_shards(int) | 用于分布式场景，将数据分为多少份，与`shard_id`配合使用 |
| shard_id(int) | 用于分布式场景，取第几份数据(0~n-1,n为设置的`num_shards`)，与`num_shards`配合使用 |
| num_parallel_workers(int) | 并行配置的线程数 |

举个例子：

```python
import mindspore.dataset as ds

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32,32,3), num_classes=10, base_seed=0)
print(dataset.get_dataset_size())
# 1000

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32,32,3), num_classes=10, base_seed=0, num_samples=3)
print(dataset.get_dataset_size())
# 3

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32,32,3), num_classes=10, base_seed=0,
                              num_shards=8, shard_id=0)
print(dataset.get_dataset_size())
# 1000 / 8 = 125
```

### 数据处理及增强

MindSpore的dataset对象使用map接口进行数据增强。我们一起来看下[map接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.map)：

```text
map(operations, input_columns=None, output_columns=None, column_order=None, num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16, offload=None)
```

给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。

每个数据增强操作将数据集对象中的一个或多个数据列作为输入，将数据增强的结果输出为一个或多个数据列。 第一个数据增强操作将 input_columns 中指定的列作为输入。 如果数据增强列表中存在多个数据增强操作，则上一个数据增强的输出列将作为下一个数据增强的输入列。

最后一个数据增强的输出列的列名由 output_columns 指定，如果没有指定 output_columns ，输出列名与 input_columns 一致。

上面的介绍可能比较繁琐，简单来说 `map` 就是在数据集的某些列上做 `operations` 里规定的操作。这里的`operations`可以是MindSpore提供的数据增强操作:
[audio](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.audio.html)，[text](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.text.html)，[vision](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.vision.html)，[通用](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.transforms.html)。
也可以是python的方法，里面可以用 opencv，PIL，pandas 等一些三方的方法，和数据集加载一样，**不要使用MindSpore的算子**。

MindSpore也提供了一些常用的随机增强方法：[自动数据增强](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/augment.html)，在具体使用数据增强时最好先阅读[数据处理性能优化](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/optimize.html)，按照推荐的顺序进行处理。

在数据增强结束后，可以使用batch算子将数据集中连续 batch_size 条数据合并为一个批处理数据。详情请参考[batch](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.batch)。其中需要注意以下 `drop_remainder` 这个参数，当训练的时候需要设置成True，推理时设置成False。

```python
import mindspore.dataset as ds

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32,32,3), num_classes=10, base_seed=0)\
    .batch(32, drop_remainder=True)
print(dataset.get_dataset_size())
# 1000 // 32 = 31

dataset = ds.FakeImageDataset(num_images=1000, image_size=(32,32,3), num_classes=10, base_seed=0)\
    .batch(32, drop_remainder=False)
print(dataset.get_dataset_size())
# ceil(1000 / 32) = 32
```

batch算子也可以使用一些batch内的增强操作，详情可参考[YOLOv3](https://gitee.com/mindspore/models/blob/master/official/cv/yolov3_darknet53/src/yolo_dataset.py#L177)。

## 数据迭代

MindSpore的数据对象有以下几种方式迭代获取。

### [create_dict_iterator](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.create_dict_iterator): 基于数据集对象创建迭代器，输出的数据为字典类型。

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)
iterator = dataset.create_dict_iterator()
for data_dict in iterator:
    for name in data_dict.keys():
        print(name, data_dict[name].shape)
    print("="*20)

# image (10, 32, 32, 3)
# label (10,)
# ====================
# image (10, 32, 32, 3)
# label (10,)
# ====================
```

### [create_tuple_iterator](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset.create_tuple_iterator)

基于数据集对象创建迭代器，输出数据为 numpy.ndarray 组成的列表。

可以通过参数 columns 指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)
iterator = dataset.create_tuple_iterator()
for data_tuple in iterator:
    for data in data_tuple:
        print(data.shape)
    print("="*20)

# (10, 32, 32, 3)
# (10,)
# ====================
# (10, 32, 32, 3)
# (10,)
# ====================
```

### 直接遍历dataset对象

> 注意这种写法在遍历完一次epoch后不会shuffle，在训练时这样使用可能会影响精度，训练时需要直接数据迭代时建议使用上面的两种方法。

```python
import mindspore.dataset as ds
dataset = ds.FakeImageDataset(num_images=20, image_size=(32, 32, 3), num_classes=10, base_seed=0)
dataset = dataset.batch(10, drop_remainder=True)

for data in dataset:
    for data in data:
        print(data.shape)
    print("="*20)

# (10, 32, 32, 3)
# (10,)
# ====================
# (10, 32, 32, 3)
# (10,)
# ====================
```

其中后两种在数据读入顺序和网络需要的顺序一致时，可以直接使用：

```python
for data in dataset:
    loss = net(*data)
```
