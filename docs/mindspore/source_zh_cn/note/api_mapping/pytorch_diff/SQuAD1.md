# 比较与torchtext.datasets.SQuAD1的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SQuAD1.md)

## torchtext.datasets.SQuAD1

```python
class torchtext.datasets.SQuAD1(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'dev'))
```

更多内容详见[torchtext.datasets.SQuAD1](https://pytorch.org/text/0.9.0/datasets.html#torchtext.datasets.SQuAD1)。

## mindspore.dataset.SQuADDataset

```python
class mindspore.dataset.SQuADDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.SQuADDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.SQuADDataset.html#mindspore.dataset.SQuADDataset)。

## 差异对比

PyTorch：读取SQuAD1数据集。

MindSpore：读取SQuAD1数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | split      | usage    |- |
|     | 参数3 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数4 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数5 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数6 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数7 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数8 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# PyTorch
import torchtext.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.SQuAD1(root, split=('train', 'dev'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download SQuAD1 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train-v1.1.json
#      ├── dev-v1.1.json
root = "/path/to/dataset_directory/"
ms_dataloader = ds.SQuADDataset(root, usage='all')
```
