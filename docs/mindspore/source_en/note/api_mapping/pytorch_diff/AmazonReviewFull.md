# Differences with torchtext.datasets.AmazonReviewFull

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AmazonReviewFull.md)

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

For more information, see [mindspore.dataset.AmazonReviewDataset](https://mindspore.cn/docs/en/r2.2/api_python/dataset/mindspore.dataset.AmazonReviewDataset.html#mindspore.dataset.AmazonReviewDataset).

## Differences

PyTorch: Read the AmazonReviewFull dataset.

MindSpore: Read the AmazonReviewFull dataset. Download dataset from web is not supported.
| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    |- |
|     | Parameter3 | -    | num_samples | The number of images to be included in the dataset |
|     | Parameter4 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter5 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter6 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter7 | -    | shard_id | The shard ID within num_shards |
|     | Parameter8 | -    | cache | Use tensor caching service to speed up dataset processing |

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
