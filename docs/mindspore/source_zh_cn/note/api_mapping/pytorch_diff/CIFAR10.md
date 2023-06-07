# 比较与torchvision.datasets.CIFAR10的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CIFAR10.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.datasets.CIFAR10

```python
class torchvision.datasets.CIFAR10(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False)
```

更多内容详见[torchvision.datasets.CIFAR10](https://pytorch.org/vision/0.14/generated/torchvision.datasets.CIFAR10.html)。

## mindspore.dataset.Cifar10Dataset

```python
class mindspore.dataset.Cifar10Dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.Cifar10Dataset](https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset)。

## 差异对比

PyTorch：读取CIFAR-10数据集。将image和label的变换操作集成在参数中。

MindSpore：读取CIFAR-10数据集，不支持下载，对image和label的变换需要使用`mindspore.dataset.map`操作。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | train      | -    | 指定是否为训练集，MindSpore通过参数`usage`支持 |
|     | 参数3 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数4 | target_transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数5 | download    | -   | MindSpore不支持 |
|     | 参数6 | -    | usage | 指定数据集的子集 |
|     | 参数7 | -    | num_samples  | 指定从数据集中读取的样本数 |
|     | 参数8 | -    | num_parallel_workers  | 指定读取数据的工作线程数 |
|     | 参数9 | -    | shuffle | 指定是否混洗数据集 |
|     | 参数10 | -    | sampler | 指定从数据集中选取样本的采样器 |
|     | 参数11 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数12 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数13 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# Torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.CIFAR10(root, train=True, transform=T.RandomCrop((28, 28)))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

root = "/path/to/dataset_directory/"
ms_dataloader = ds.Cifar10Dataset(root, usage='train')
ms_dataloader = ms_dataloader.map(vision.RandomCrop((28, 28)), ["image"])
```
