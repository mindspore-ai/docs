# 比较与torchvision.transforms.ConvertImageDtype的功能差异

## torchvision.transforms.ConvertImageDtype

```python
class torchvision.transforms.ConvertImageDtype(
    dtype: torch.dtype
    )
```

更多内容详见[torchvision.transforms.ConvertImageDtype](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ConvertImageDtype)。

## mindspore.dataset.vision.py_transforms.ToType(output_type)

```python
class mindspore.dataset.vision.py_transforms.ToType(
    output_type
    )
```

更多内容详见[mindspore.dataset.vision.py_transforms.ToType](https://mindspore.cn/docs/api/zh-CN/r1.3/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToType.html#mindspore.dataset.vision.py_transforms.ToType)。

## 使用方式

PyTorch：将张量图像转换为给定的数据类型并相应缩放值，此算子不支持PIL图像。

MindSpore：将输入的numpy.ndarray图像转换为所需的数据类型。

## 代码示例

```python
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import mindspore.dataset.vision.py_transforms as py_vision

# In MindSpore, ToType act through map operation.

coco_dataset_dir = "/path/to/coco/testCOCO/train"
coco_annotation_file = "/path/to/coco/testCOCO/annotations/train.json"

dataset = ds.CocoDataset(
    dataset_dir=coco_dataset_dir,
    annotation_file=coco_annotation_file,
    task='Detection')
transforms_list =py_vision.Compose(
    [py_vision.Decode(),
    py_vision.ToTensor(),
    py_vision.ToType(np.float32)])
dataset  = dataset.map(operations=transforms_list, input_columns="image")

for item in dataset:
    print(len(item[0]))
    break
# Out:
#  3

# In torch, ConvertImageDtype act through Sequential operation.
coco_dataset_dir = "/path/to/coco_dataset_directory/images"
coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"

#Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
dataset = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)

for epoch in range(1):
    for i, batch in enumerate(dataloader):
        transformers = T.Compose([transforms.ConvertImageDtype(torch.float)])
        real_a = batch[0]
        real_a = transformers(real_a)
        print(real_a.shape)
        print(real_a.dtype)
# Out:
# loading annotations into memory...
# Done (t=0.00s)
# creating index...
# index created!
# torch.Size([1, 3, 561, 595])
# torch.float32
# ...
```
