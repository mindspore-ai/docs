# Function differences with torchvision.transforms.ConvertImageDtype

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TypeCast.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.ConvertImageDtype

```python
class torchvision.transforms.ConvertImageDtype(
    dtype: torch.dtype
    )
```

For more information, see [torchvision.transforms.ConvertImageDtype](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ConvertImageDtype).

## mindspore.dataset.transforms.TypeCast(output_type)

```python
class mindspore.dataset.transforms.TypeCast(
    output_type
    )
```

For more information, see [mindspore.dataset.transforms.TypeCast](https://mindspore.cn/docs/en/r1.9/api_python/dataset_transforms/mindspore.dataset.transforms.TypeCast.html#mindspore.dataset.transforms.TypeCast).

## Differences

PyTorch: Convert a tensor image to the given dtype and scale the values accordingly. This function does not support PIL Image.

MindSpore: Convert the input numpy.ndarray image to the desired dtype.

## Code Example

```python
import numpy as np
import mindspore.dataset as ds
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# In MindSpore, TypeCast act through map operation.

coco_dataset_dir = "/path/to/coco/testCOCO/train"
coco_annotation_file = "/path/to/coco/testCOCO/annotations/train.json"

dataset = ds.CocoDataset(
    dataset_dir=coco_dataset_dir,
    annotation_file=coco_annotation_file,
    task='Detection')
transforms_list = transforms.Compose(
    [vision.Decode(to_pil=True),
    vision.ToTensor(),
    transforms.TypeCast(np.float32)])
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
dataset = datasets.CocoDetection(coco_dataset_dir, coco_annotation_file, transform=T.ToTensor())
dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)
for epoch in range(1):
    for i, batch in enumerate(dataloader):
        transformers = T.Compose([T.ConvertImageDtype(torch.float)])
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
