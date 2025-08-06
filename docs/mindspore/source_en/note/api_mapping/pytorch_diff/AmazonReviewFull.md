# Differences with torchtext.datasets.AmazonReviewFull

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AmazonReviewFull.md)

## torchtext.datasets.AmazonReviewFull

```python
class torchtext.datasets.AmazonReviewFull(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'test'))
```

For more information, see [torchtext.datasets.AmazonReviewFull](https://pytorch.org/text/0.9.0/datasets.html#amazonreviewfull).

## mindspore.dataset.AmazonReviewDataset

```python
class mindspore.dataset.AmazonReviewDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.AmazonReviewDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.AmazonReviewDataset.html#mindspore.dataset.AmazonReviewDataset).

## Differences

PyTorch: Read the AmazonReviewFull dataset.

MindSpore: Read the AmazonReviewFull dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameters | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    |- |
|     | Parameter3 | -    | num_samples | Specify the number of samples to read from the dataset |
|     | Parameter4 | -    | num_parallel_workers | Specify the number of worker threads to read from the dataset|
|     | Parameter5 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter6 | -    | num_shards | Number of shards that the dataset will be divided into during the distributed training |
|     | Parameter7 | -    | shard_id | Specify the shard ID to be used for distributed training |
|     | Parameter8 | -    | cache | Specify single node data caching service |

## Code Example

```python
# PyTorch
import torchtext.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.AmazonReviewFull(root, split=('train', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download AmazonReviewFull dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train.csv
#      ├── test.csv
#      ├── readme.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.AmazonReviewDataset(root, usage='all')
```
