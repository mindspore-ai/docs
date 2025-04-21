# 比较与torchvision.datasets.CelebA的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CelebA.md)

## torchvision.datasets.CelebA

```python
class torchvision.datasets.CelebA(
    root: str,
    split: str = 'train',
    target_type: Union[List[str], str] = 'attr',
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False)
```

更多内容详见[torchvision.datasets.CelebA](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.CelebA)。

## mindspore.dataset.CelebADataset

```python
class mindspore.dataset.CelebADataset(
    dataset_dir,
    num_parallel_workers=None,
    shuffle=None,
    usage='all',
    sampler=None,
    decode=False,
    extensions=None,
    num_samples=None,
    num_shards=None,
    shard_id=None,
    cache=None,
    decrypt=None)
```

更多内容详见[mindspore.dataset.CelebADataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.CelebADataset.html#mindspore.dataset.CelebADataset)。

## 差异对比

PyTorch：读取CelebA（CelebFaces Attributes）数据集。将image和label的变换操作集成在参数中。

MindSpore：读取CelebA（CelebFaces Attributes）数据集，不支持下载，对image和label的变换需要使用`mindspore.dataset.map`操作。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | split      | usage    |- |
|     | 参数3 | target_type      | -    | - |
|     | 参数4 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数5 | target_transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数6 | download    | -   | MindSpore不支持 |
|     | 参数7 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数8 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数9 | -    | sampler  | 指定采样器 |
|     | 参数10 | -    | decode | 指定是否对图像进行解码 |
|     | 参数11 | -    | extensions | 指定读取文件的扩展名 |
|     | 参数12 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数13 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数14 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数15 | -    | cache | 指定单节点数据缓存服务 |
|     | 参数16 | -    | decrypt | 指定图像解密函数 |

## 代码示例

```python
# PyTorch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.CelebA(root, split='train', target_type="attr", transform=T.ToTensor(), download=True)
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Download CelebA dataset files, unzip the img_align_celeba.zip and put list_attr_celeba.txt together like
# .
# └── /path/to/dataset_directory/
#      ├── list_attr_celeba.txt
#      ├── 000001.jpg
#      ├── 000002.jpg
#      ├── 000003.jpg
#      ├── ...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.CelebADataset(root, usage='train', decode=True)
ms_dataloader = ms_dataloader.map(vision.ToTensor(), ["image"])
```
