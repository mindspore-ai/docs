# 比较与torchvision.datasets.MNIST的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MNIST.md)

## torchvision.datasets.MNIST

```python
class torchvision.datasets.MNIST(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False)
```

更多内容详见[torchvision.datasets.MNIST](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.MNIST)。

## mindspore.dataset.MnistDataset

```python
class mindspore.dataset.MnistDataset(
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

更多内容详见[mindspore.dataset.MnistDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset)。

## 差异对比

PyTorch：读取MNIST数据集。将image和label的变换操作集成在参数中。

MindSpore：读取MNIST数据集，不支持下载，对image和label的变换需要使用`mindspore.dataset.map`操作。

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
# PyTorch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.MNIST(root, train=False, transform=T.Resize((32, 32)), download=True)
dataloader = DataLoader(dataset, batch_size=32)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Download the dataset files, unzip into the following structure
# .
# └── "/path/to/dataset_directory/"
#      ├── t10k-images-idx3-ubyte
#      ├── t10k-labels-idx1-ubyte
#      ├── train-images-idx3-ubyte
#      └── train-labels-idx1-ubyte
root = "/path/to/dataset_directory/"
ms_dataloader = ds.Cifar10Dataset(root, usage='test')
ms_dataloader = ms_dataloader.map(vision.Resize((32, 32)), ["image"])
ms_dataloader = ms_dataloader.batch(32)
```
