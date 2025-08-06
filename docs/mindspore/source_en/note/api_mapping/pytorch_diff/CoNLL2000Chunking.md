# Differences with torchtext.datasets.CoNLL2000Chunking

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CoNLL2000Chunking.md)

## torchtext.datasets.CoNLL2000Chunking

```python
class torchtext.datasets.CoNLL2000Chunking(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'test'))
```

For more information, see [torchtext.datasets.CoNLL2000Chunking](https://pytorch.org/text/0.9.0/datasets.html#conll2000chunking).

## mindspore.dataset.CoNLL2000Dataset

```python
class mindspore.dataset.CoNLL2000Dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.CoNLL2000Dataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.CoNLL2000Dataset.html#mindspore.dataset.CoNLL2000Dataset).

## Differences

PyTorch: Read the CoNLL2000 dataset.

MindSpore: Read the CoNLL2000 dataset. Downloading dataset from web is not supported.

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
dataset = datasets.CoNLL2000Chunking(root, split=('train', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download CoNLL2000 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train.txt
#      ├── test.txt
#      ├── readme.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.CoNLL2000Dataset(root, usage='all')
```
