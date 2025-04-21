# 比较与torchvision.datasets.ImageFolder的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ImageFolder.md)

## torchvision.datasets.ImageFolder

```python
class torchvision.datasets.ImageFolder(
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Union[Callable, NoneType] = None,
    loader: Optional[Callable] = None,
    is_valid_file: bool = None)
```

更多内容详见[torchvision.datasets.ImageFolder](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.ImageFolder)。

## mindspore.dataset.ImageFolderDataset

```python
class mindspore.dataset.ImageFolderDataset(
    dataset_dir,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    extensions=None,
    class_indexing=None,
    decode=False,
    num_shards=None,
    shard_id=None,
    cache=None,
    decrypt=None)
```

更多内容详见[mindspore.dataset.ImageFolderDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.ImageFolderDataset.html#mindspore.dataset.ImageFolderDataset)。

## 差异对比

PyTorch：从树状结构的文件目录中读取图片构建源数据集，将image和label的变换操作集成在参数中。支持自定义读取方法。

MindSpore：从树状结构的文件目录中读取图片构建源数据集，对image和label的变换需要使用`mindspore.dataset.map`操作。不支持自定义读取方法。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数3 | target_transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数4 | loader    | -   | MindSpore不支持 |
|     | 参数5 | is_valid_file    | -   | MindSpore不支持 |
|     | 参数6 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数7 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数8 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数9 | -    | sampler  | 指定采样器 |
|     | 参数10 | -    | extensions | 指定读取文件的扩展名 |
|     | 参数11 | -    | class_indexing | 指定文件夹名称到label索引的映射 |
|     | 参数12 | -    | decode | 指定是否对图像进行解码 |
|     | 参数13 | -    | num_shards | 指定分布式训练时，将数据集进行划分的分片数 |
|     | 参数14 | -    | shard_id | 指定分布式训练时，使用的分片ID号 |
|     | 参数15 | -    | cache | 指定单节点数据缓存服务 |
|     | 参数16 | -    | decrypt | 指定图像解密函数 |

## 代码示例

假设文件目录具有如下树状结构：

```text
imageset/
    ├── cat
    │   ├── cat_0.jpg
    │   ├── cat_1.jpg
    │   └── cat_2.jpg
    ├── fish
    │   ├── fish_0.jpg
    │   ├── fish_1.jpg
    │   ├── fish_2.jpg
    │   └── fish_3.jpg
    ├── fruits
    │   ├── fruits_0.jpg
    │   ├── fruits_1.jpg
    │   └── fruits_2.jpg
    ├── plane
    │   ├── plane_0.jpg
    │   ├── plane_1.jpg
    │   └── plane_2.jpg
    └── tree
        ├── tree_0.jpg
        ├── tree_1.jpg
        └── tree_2.jpg
```

```python
# Torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/imageset/"
dataset = datasets.ImageFolder(root, transform=T.RandomCrop((256, 256)))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

root = "/path/to/imageset/"
ms_dataloader = ds.ImageFolderDataset(root, decode=True)
ms_dataloader = ms_dataloader.map(vision.RandomCrop((256, 256)), ["image"])
```
