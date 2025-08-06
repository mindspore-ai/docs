# Differences with torchvision.transforms.ToTensor

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ToTensor.md)

## torchvision.transforms.ToTensor

```python
class torchvision.transforms.ToTensor
```

For more information, see [torchvision.transforms.ToTensor](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToTensor).

## mindspore.dataset.vision.ToTensor

```python
class mindspore.dataset.vision.ToTensor(
    output_type=np.float32
    )
```

For more information, see [mindspore.dataset.vision.ToTensor](https://mindspore.cn/docs/en/br_base/api_python/dataset_vision/mindspore.dataset.vision.ToTensor.html#mindspore.dataset.vision.ToTensor).

## Differences

PyTorch: Convert the PIL Image or Numpy array to tensor. The input Numpy array is usually in the format of <H, W, C> and the value is in the range of [0, 255], and the output is the torch tensor with format of <C, H, W > in the range of [0.0, 1.0].

MindSpore: The input is an image of PIL type or a Numpy array with a value in the range of [0, 255] in the format of <H, W, C>, and the output is a Numpy array in the range of [0.0, 1.0] with the format of <C, H, W>; it is equivalent to two operations of channel conversion and pixel value normalization on the original input image.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameters 1  | -    | output_type  | Specify the data type of output numpy array |

## Code Example

```python
import numpy as np
from PIL import Image
from download import download
from torchvision import transforms
import mindspore.dataset.vision as vision

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
img = Image.open('flamingos.jpg')

# In MindSpore, ToTensor convert PIL Image into numpy array.
to_tensor = vision.ToTensor()
img_data = to_tensor(img)
print("img_data shape:", img_data.shape)
# img_data shape: (3, 292, 471)
print("img_data type:", type(img_data))
# img_data type: <class 'numpy.ndarray'>

# In torch, ToTensor transforms the input to tensor.
image_transform = transforms.Compose([transforms.ToTensor()])
img_data = image_transform(img)
print("img_data shape:", img_data.shape)
# img_data shape: torch.Size([3, 292, 471])
print("img_data type:", type(img_data))
# img_data type: <class 'torch.Tensor'>
```
