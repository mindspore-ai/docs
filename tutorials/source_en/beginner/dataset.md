# Data Processing

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/dataset.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

Data is the foundation of deep learning, and inputting the high-quality data plays an active role in the entire deep neural network.

[mindspore.dataset](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.html) provides a loading interface for some commonly used datasets and standard format datasets, enabling users to quickly perform data processing operations. For the image datasets, users can use `mindvision.dataset` to load and process datasets. This chapter first describes how to load and process a CIFAR-10 dataset by using the `mindvision.dataset.Cifar10` interface, and then describes how to use `mindspore.dataset.GeneratorDataset` to implement custom dataset loading.

> `mindvision.dataset`is a dataset interface developed on the basis of `mindspore.dataset`. In addition to providing dataset loading capabilities, `mindvision.dataset` further provides dataset download capabilities, data processing, and data enhancement capabilities.

## Data Process

In the network training and inference process, the raw data is generally stored in a disk or database. The raw data needs to be read into the memory space through the data loading step, converted into a framework-common tensor format, and then mapped to a more easy-to-learn space through the data processing and augmentation steps. While the number of samples and generalization is increased, the data finally enters the network for calculation.

The overall process is shown in the following figure:

![dataset_pipeline](https://gitee.com/mindspore/docs/raw/master/tutorials/source_zh_cn/beginner/images/dataset_pipeline.png)

### Dataset

A dataset is a collection of samples, and a row of a dataset is a sample that contains one or more features, and may further contain a label. The dataset needs to meet certain specification requirements to make it easier to evaluate the effectiveness of the model.

Dataset supports multiple format datasets, including MindRecord, a MindSpore self-developed data format, commonly used public image datasets and text datasets, user-defined datasets, etc.

### Dataset Loading

Dataset loading allows the model to be continuously acquired for training during training. Dataset provides corresponding classes for a variety of commonly used datasets to load datasets. For data files in different storage formats, Dataset also has corresponding classes for data loading.

Dataset provides multiple uses of the sampler (Sampler), and the sampler is responsible for generating the read index sequence. The Dataset is responsible for reading the corresponding data according to the index, helping users to sample the dataset in different forms to meet the training needs, and solving problems such as the data set is too large or the sample class distribution is uneven.

> It should be noted that the sampler is responsible for performing filter and reorder operations on the sample, not performing the Batch operation.

### Data processing

After the Dataset loads the data into the memory, the data is organized in a Tensor form. Tensor is also a basic data structure in data augmentation operations.

## Loading the Dataset

In the following example, the CIFAR-10 dataset is loaded through the `mindvision.dataset.Cifar10` interface. The CIFAR-10 dataset has a total of 60,000 32*32 color images, which are divided into 10 categories, each with 6,000 maps, and a total of 50,000 training pictures and 10,000 test pictures in the dataset. `Cifar10` interface provides CIFAR-10 dataset download and load capabilities.

![cifar10](https://gitee.com/mindspore/docs/raw/master/tutorials/source_zh_cn/beginner/images/cifar10.jpg)

- `path`: The location of the dataset root directory.
- `split`: Training, testing or inferencing of the dataset, optionally `train`，`test` or `infer`, `train` by default.
- `download`: Whether to download the dataset. When `ture` is set, if the dataset does not exist, you can download and extract the dataset, `False` by default.

```python
from mindvision.dataset import Cifar10

# Dataset root directory
data_dir = "./datasets"

# Download, extract and load the CIFAR-10 training dataset
dataset = Cifar10(path=data_dir, split='train', batch_size=6, resize=32, download=True)
dataset = dataset.run()
```

The directory structures of the CIFAR-10 dataset files are as follows:

```text
datasets/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
└── cifar-10-python.tar.gz
```

## Iterating Dataset

You can use `create_dict_iterator`  interface to create a data iterator to iteratively access data. The data type of the access is `Tensor` by default, and if `output_numpy=True` is set, the data type of the access is `Numpy`.

The following shows the corresponding access data type, and the image shapes and labels.

```python
data = next(dataset.create_dict_iterator())
print(f"Data type:{type(data['image'])}\nImage shape: {data['image'].shape}, Label: {data['label']}")

data = next(dataset.create_dict_iterator(output_numpy=True))
print(f"Data type:{type(data['image'])}\nImage shape: {data['image'].shape}, Label: {data['label']}")
```

```text
Data type:<class 'mindspore.common.tensor.Tensor'>
Image shape: (6, 3, 32, 32), Label: [7 1 2 8 7 8]
Data type:<class 'numpy.ndarray'>
Image shape: (6, 3, 32, 32), Label: [8 0 0 2 6 1]
```

## Data Processing and Augmentation

### Data Processing

`mindvision.dataset.Cifar10` interface provides data processing capbilities. The data can be processed by simply setting the corresponding attributes.

- `shuffle`: Whether to disrupt the order of the datasets, when `True` is  set, the order of the datasets is disturbed, `False` by default .
- `batch_size`: The number of data contained in each group. The `batch_size=2` contains 2 data per group, and the default size of the `batch_size` value is 32.
- `repeat_num`: For the number of duplicate datasets. `repeat_num=1` is a dataset, and the default value of the `repeat_num`  is 1.

```python
import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset.vision.c_transforms as transforms

trans = [transforms.HWC2CHW()]
dataset = Cifar10(data_dir, batch_size=6, resize=32, repeat_num=1, shuffle=True, transform=trans)
data = dataset.run()
data = next(data.create_dict_iterator())

images = data["image"].asnumpy()
labels = data["label"].asnumpy()
print(f"Image shape: {images.shape}, Label: {labels}")

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    image_trans = np.transpose(images[i-1], (1, 2, 0))
    plt.title(f"{dataset.index2label[labels[i-1]]}")
    plt.imshow(image_trans, interpolation="None")
plt.show()
```

```text
Image shape: (6, 3, 32, 32), Label: [9 3 8 9 6 8]
```

### Data Augmentation

Problems such as too small amount of data or single sample scene will affect the training effect of the model, and users can expand the diversity of samples through data augmentation operations to improve the generalization ability of the model. The `mindvision.dataset.Cifar10` interface uses the default data augmentation feature, which allows users to perform data augmentation operations by setting attribute `transform` and `target_transform`.

- `transform`: augment dataset image data.
- `target_transform`: process the dataset label data.

This section describes data augmentation of the CIFAR-10 dataset by using operators in the `mindspore.dataset.vision .c_transforms` module.

```python
import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset.vision.c_transforms as transforms

# Image augmentation
trans = [
    transforms.RandomCrop((32, 32), (4, 4, 4, 4)), # Automatic cropping of images
    transforms.RandomHorizontalFlip(prob=0.5), # Flip the image randomly and horizontally
    transforms.HWC2CHW(), # Convert (h, w, c) to (c, h, w)
]

dataset = Cifar10(data_dir, batch_size=6, resize=32, transform=trans)
data = dataset.run()
data = next(data.create_dict_iterator())
images = data["image"].asnumpy()
labels = data["label"].asnumpy()
print(f"Image shape: {images.shape}, Label: {labels}")

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    image_trans = np.transpose(images[i-1], (1, 2, 0))
    plt.title(f"{dataset.index2label[labels[i-1]]}")
    plt.imshow(image_trans, interpolation="None")
plt.show()
```

```text
Image shape: (6, 3, 32, 32), Label: [7 6 7 4 5 3]
```

