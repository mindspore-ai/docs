# Differences with torchtext.datasets.IMDB

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/IMDB.md)

## torchtext.datasets.IMDB

```python
class torchtext.datasets.IMDB(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'test'))
```

For more information, see [torchtext.datasets.IMDB](https://pytorch.org/text/0.9.0/datasets.html#imdb).

## mindspore.dataset.IMDBDataset

```python
class mindspore.dataset.IMDBDataset(
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

For more information, see [mindspore.dataset.IMDBDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.IMDBDataset.html#mindspore.dataset.IMDBDataset).

## Differences

PyTorch: Read the IMDB dataset.

MindSpore: Read the IMDB dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    |- |
|     | Parameter3 | -    | num_samples | The number of images to be included in the dataset |
|     | Parameter4 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter5 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter6 | -    | sampler  | Specify the sampler |
|     | Parameter7 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter8 | -    | shard_id | The shard ID within num_shards |
|     | Parameter9 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchtext.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.IMDB(root, split=('train', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download IMDB dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── train
#      │    ├── pos
#      │    │    ├── 0_9.txt
#      │    │    ├── 1_7.txt
#      │    │    ├── ...
#      │    ├── neg
#      │    │    ├── 0_3.txt
#      │    │    ├── 1_1.txt
#      │    │    ├── ...
#      ├── test
#      │    ├── pos
#      │    │    ├── 0_10.txt
#      │    │    ├── 1_10.txt
#      │    │    ├── ...
#      │    ├── neg
#      │    │    ├── 0_2.txt
#      │    │    ├── 1_3.txt
#      │    │    ├── ...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.IMDBDataset(root, usage='all')
```
