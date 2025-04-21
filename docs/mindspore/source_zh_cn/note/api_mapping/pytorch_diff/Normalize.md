# 比较与torchvision.transforms.Normalize的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Normalize.md)

## torchvision.transforms.Normalize

```python
class torchvision.transforms.Normalize(mean, std, inplace=False)
```

更多内容详见[torchvision.transforms.Normalize](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.Normalize)。

## mindspore.dataset.vision.Normalize

```python
class mindspore.dataset.vision.Normalize(mean, std, is_hwc=True)
```

更多内容详见[mindspore.dataset.vision.Normalize](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.Normalize.html)。

## 差异对比

PyTorch：根据均值和标准差对输入图像进行归一化，不支持指定图像的格式。

MindSpore：根据均值和标准差对输入图像进行归一化，不支持原地修改。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | mean   | mean     | - |
|     | 参数2 | std    |std   | - |
|     | 参数3 | inplace | -   | 是否对Tensor进行原地修改 |
|     | 参数4 | -   | is_hwc    | 指定图像的格式是否为HWC或CHW格式 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

normalize = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
to_tensor = T.ToTensor()
img_torch = T.Compose([to_tensor, normalize])((orig_img))
print(img_torch.shape)
# Torch tensor is in C,H,W format
# Out: torch.Size([3, 292, 471])

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

normalize = vision.Normalize(mean=[0, 0, 0], std=[1, 1, 1], is_hwc=False)
to_tensor = vision.ToTensor()
img_ms = transforms.Compose([to_tensor, normalize])((orig_img))
print(img_ms[0].shape)
# vision.ToTensor change the format from HWC to CHW, so normalize have to specify `is_hwc=False`
# Out: (3, 292, 471)
```
