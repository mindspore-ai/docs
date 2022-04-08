# 比较与torchvision.transforms.ToTensor的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ToTensor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.ToTensor

```python
class torchvision.transforms.ToTensor
```

更多内容详见[torchvision.transforms.ToTensor](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ToTensor)。

## mindspore.dataset.vision.py_transforms.ToTensor

```python
class mindspore.dataset.vision.py_transforms.ToTensor(
    output_type=np.float32
    )
```

更多内容详见[mindspore.dataset.vision.py_transforms.ToTensor](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToTensor.html#mindspore.dataset.vision.py_transforms.ToTensor)。

## 使用方式

PyTorch：将PIL类型的Image或numpy 数组转换为 torch中的Tensor, 输入的numpy数组通常是<H, W, C>格式且取值在[0, 255]范围，输出是<C, H, W>格式且取值在[0.0, 1.0]的torch Tensor。

MindSpore：输入为PIL类型的图像或<H, W, C>格式且取值在[0, 255]范围内的numpy数组，输出为[0.0, 1.0]范围内且具有<C, H, W>格式的numpy数组；等同于在原始输入图像上做了通道转换及像素值归一化两种操作。

## 代码示例

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
