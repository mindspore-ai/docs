# 比较与torchtext.datasets.YelpReviewPolarity的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/YelpReviewPolarity.md)

## torchtext.datasets.YelpReviewPolarity

```python
class torchtext.datasets.YelpReviewPolarity(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'test'))
```

更多内容详见[torchtext.datasets.YelpReviewPolarity](https://pytorch.org/text/0.9.0/datasets.html#yelpreviewpolarity)。

## mindspore.dataset.YelpReviewDataset

```python
class mindspore.dataset.YelpReviewDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.YelpReviewDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.YelpReviewDataset.html#mindspore.dataset.YelpReviewDataset)。

## 差异对比

PyTorch：读取YelpReviewPolarity数据集。

MindSpore：读取YelpReviewPolarity数据集，不支持下载。

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
dataset = datasets.YelpReviewPolarity(root, split=('train', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download YelpReviewPolarity dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train.csv
#      ├── test.csv
#      ├── readme.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.YelpReviewDataset(root, usage='all')
```
