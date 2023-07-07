# 比较与torchtext.datasets.AG_NEWS的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AGNEWS.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchtext.datasets.AG_NEWS

```python
class torchtext.datasets.AG_NEWS(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'test'))
```

更多内容详见[torchtext.datasets.AG_NEWS](https://pytorch.org/text/0.9.0/datasets.html#ag-news)。

## mindspore.dataset.AGNewsDataset

```python
class mindspore.dataset.AGNewsDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.AGNewsDataset](https://mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.AGNewsDataset.html#mindspore.dataset.AGNewsDataset)。

## 差异对比

PyTorch：读取AG News数据集。

MindSpore：读取AG News数据集，不支持下载。

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
dataset = datasets.AG_NEWS(root, split=('train', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download AG News dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── classes.txt
#      ├── train.csv
#      ├── test.csv
#      ├── readme.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.AGNewsDataset(root, usage='all')
```
