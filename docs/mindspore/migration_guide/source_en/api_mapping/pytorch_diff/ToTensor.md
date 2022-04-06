# Function differences with torchvision.transforms.ToTensor

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/ToTensor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.ToTensor

```python
class torchvision.transforms.ToTensor
```

For more information, see [torchvision.transforms.ToTensor](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ToTensor).

## mindspore.dataset.vision.py_transforms.ToTensor

```python
class mindspore.dataset.vision.py_transforms.ToTensor(
    output_type=np.float32
    )
```

For more information, see [mindspore.dataset.vision.py_transforms.ToTensor](https://mindspore.cn/docs/api/en/master/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToTensor.html#mindspore.dataset.vision.py_transforms.ToTensor).

## Differences

PyTorch: Convert the PIL Image or numpy array to tensor. The input numpy array is usually in the format of <H, W, C> and the value is in the range of [0, 255], and the output is <C, H, W > Torch Tensor with format and value in [0.0, 1.0].

MindSpore: The input is an image of PIL type or a numpy array with a value in the range of [0, 255] in the format of <H, W, C>, and the output is in the range of [0.0, 1.0] with <C, H, W> Format numpy array; it is equivalent to two operations of channel conversion and pixel value normalization on the original input image.

## Code Example

```python
import numpy as np
from PIL import Image
from torchvision import transforms
from mindspore.dataset import py_transforms
import mindspore.dataset.vision.py_transforms as py_vision

# In MindSpore, ToTensor convert PIL Image into numpy array.
img_path =  "/path/to/test/1.jpg"

img = Image.open(img_path)
to_tensor = py_vision.ToTensor()
img_data = to_tensor(img)
print("img_data type:", type(img_data))
print("img_data dtype:", img_data.dtype)

# Out:
#img_data type: <class 'numpy.ndarray'>
#img_data dtype: float32

# In torch, ToTensor transforms the input to tensor.
img_path = "/path/to/test/1.jpg"

image_transform = transforms.Compose([transforms.ToTensor()])
img = np.array(Image.open(img_path))
img_data = image_transform(img)
print(img_data.shape)
# Out:
# torch.Size([3, 2268, 4032])
```
