# Differences with torchtext.datasets.UDPOS

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/UDPOS.md)

## torchtext.datasets.UDPOS

```python
class torchtext.datasets.UDPOS(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'))
```

For more information, see [torchtext.datasets.UDPOS](https://pytorch.org/text/0.9.0/datasets.html#udpos).

## mindspore.dataset.UDPOSDataset

```python
class mindspore.dataset.UDPOSDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.UDPOSDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.UDPOSDataset.html#mindspore.dataset.UDPOSDataset).

## Differences

PyTorch: Read the UDPOS dataset.

MindSpore: Read the UDPOS dataset. Downloading dataset from web is not supported.

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
dataset = datasets.UDPOS(root, split=('train', 'valid', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download UDPOS dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── en-ud-tag.v2.dev.txt
#      ├── en-ud-tag.v2.test.txt
#      ├── en-ud-tag.v2.train.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.UDPOSDataset(root, usage='all')
```
