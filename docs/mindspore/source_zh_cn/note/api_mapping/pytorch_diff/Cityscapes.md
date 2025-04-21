# 比较与torchvision.datasets.Cityscapes的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Cityscapes.md)

## torchvision.datasets.Cityscapes

```python
class torchvision.datasets.Cityscapes(
    root: str,
    split: str,
    mode: str,
    target_type: str or list,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None
    )
```

更多内容详见[torchvision.datasets.Cityscapes](https://pytorch.org/vision/0.9/datasets.html#cityscapes)。

## mindspore.dataset.CityscapesDataset

```python
class mindspore.dataset.CityscapesDataset(
    dataset_dir,
    usage='train',
    quality_mode='fine',
    task='instance',
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    decode=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None
    )
```

更多内容详见[mindspore.dataset.CityscapesDataset](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.CityscapesDataset.html)。

## 差异对比

PyTorch：读取Cityscapes数据集。

MindSpore：读取Cityscapes数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | split      | usage    | - |
|     | 参数3 | mode    | quality_mode   | - |
|     | 参数4 | target_type    | task   | - |
|     | 参数5 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数6 | target_transform    | - | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数7 | transforms    | -  | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数8 | -    | num_samples  | 指定从数据集中读取的样本数 |
|     | 参数9 | -    | num_parallel_workers  | 指定读取数据的工作线程数 |
|     | 参数10 | -    | shuffle | 指定是否混洗数据集 |
|     | 参数11 | -    | decode | 解码读取的图片 |
|     | 参数12 | -    | sampler | 指定从数据集中选取样本的采样器 |
|     | 参数13 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数14 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数15 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# PyTorch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.Cityscapes(root, split='train', mode='fine', target_type='semantic')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Download the dataset files, unzip into the following structure
# .
# └── "/path/to/dataset_directory"
#      ├── leftImg8bit
#      |    ├── train
#      |    |    ├── aachen
#      |    |    |    ├── aachen_000000_000019_leftImg8bit.png
#      |    |    |    ├── aachen_000001_000019_leftImg8bit.png
#      |    |    |    ├── ...
#      |    |    ├── bochum
#      |    |    |    ├── ...
#      |    |    ├── ...
#      |    ├── test
#      |    |    ├── ...
#      |    ├── val
#      |    |    ├── ...
#      └── gtFine
#           ├── train
#           |    ├── aachen
#           |    |    ├── aachen_000000_000019_gtFine_color.png
#           |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
#           |    |    ├── aachen_000000_000019_gtFine_labelIds.png
#           |    |    ├── aachen_000000_000019_gtFine_polygons.json
#           |    |    ├── aachen_000001_000019_gtFine_color.png
#           |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
#           |    |    ├── aachen_000001_000019_gtFine_labelIds.png
#           |    |    ├── aachen_000001_000019_gtFine_polygons.json
#           |    |    ├── ...
#           |    ├── bochum
#           |    |    ├── ...
#           |    ├── ...
#           ├── test
#           |    ├── ...
#           └── val
#                ├── ...

root = "/path/to/dataset_directory/"
ms_dataloader = ds.CityscapesDataset(root, usage='train')
ms_dataloader = ms_dataloader.map(vision.RandomCrop((28, 28)), ["image"])
```
