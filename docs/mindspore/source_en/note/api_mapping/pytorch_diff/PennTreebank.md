# Differences with torchtext.datasets.PennTreebank

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/PennTreebank.md)

## torchtext.datasets.PennTreebank

```python
class torchtext.datasets.PennTreebank(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'))
```

For more information, see [torchtext.datasets.PennTreebank](https://pytorch.org/text/0.9.0/datasets.html#penntreebank).

## mindspore.dataset.PennTreebankDataset

```python
class mindspore.dataset.PennTreebankDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.PennTreebankDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.PennTreebankDataset.html#mindspore.dataset.PennTreebankDataset).

## Differences

PyTorch: Read the PennTreebank dataset.

MindSpore: Read the PennTreebank dataset. Downloading dataset from web is not supported.
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
dataset = datasets.PennTreebank(root, split=('train', 'valid', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download PennTreebank dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── ptb.train.txt
#      ├── ptb.test.txt
#      ├── ptb.valid.txt
root = "/path/to/dataset_directory/"
ms_dataloader = ds.PennTreebankDataset(root, usage='all')
```
