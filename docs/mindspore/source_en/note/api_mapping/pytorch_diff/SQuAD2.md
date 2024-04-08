# Differences with torchtext.datasets.SQuAD2

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SQuAD2.md)

## torchtext.datasets.SQuAD2

```python
class torchtext.datasets.SQuAD2(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'dev'))
```

For more information, see [torchtext.datasets.SQuAD2](https://pytorch.org/text/0.9.0/datasets.html#torchtext.datasets.SQuAD2).

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

For more information, see [mindspore.dataset.SQuADDataset](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/dataset/mindspore.dataset.SQuADDataset.html#mindspore.dataset.SQuADDataset).

## Differences

PyTorch: Read the SQuAD2 dataset.

MindSpore: Read the SQuAD2 dataset. Downloading dataset from web is not supported.

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
dataset = datasets.SQuAD2(root, split=('train', 'dev'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download SQuAD2 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train-v2.0.json
#      ├── dev-v2.0.json
root = "/path/to/dataset_directory/"
ms_dataloader = ds.SQuADDataset(root, usage='all')
```
