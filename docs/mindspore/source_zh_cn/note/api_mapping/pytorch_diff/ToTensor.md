# 比较与torchvision.transforms.ToTensor的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ToTensor.md)

## torchvision.transforms.ToTensor

```python
class torchvision.transforms.ToTensor
```

更多内容详见[torchvision.transforms.ToTensor](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToTensor)。

## mindspore.dataset.vision.ToTensor

```python
class mindspore.dataset.vision.ToTensor(
    output_type=np.float32
    )
```

更多内容详见[mindspore.dataset.vision.ToTensor](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.ToTensor.html#mindspore.dataset.vision.ToTensor)。

## 差异对比

PyTorch：将PIL类型的Image或Numpy数组转换为torch中的Tensor，输入的Numpy数组通常是<H, W, C>格式且取值在[0, 255]范围，输出是<C, H, W>格式且取值在[0.0, 1.0]的torch Tensor。

MindSpore：输入为PIL类型的图像或<H, W, C>格式且取值在[0, 255]范围内的Numpy数组，输出为[0.0, 1.0]范围内且具有<C, H, W>格式的Numpy数组；等同于在原始输入图像上做了通道转换及像素值归一化两种操作。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | -    | output_type   | 指定输出Numpy数组的类型 |

## 代码示例

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
