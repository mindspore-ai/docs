# Differences with torchtext.datasets.WikiText2

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/WikiText2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchtext.datasets.WikiText2

```python
class torchtext.datasets.WikiText2(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'))
```

For more information, see [torchtext.datasets.WikiText2](https://pytorch.org/text/0.9.0/datasets.html#wikitext-2).

## mindspore.dataset.WikiTextDataset

```python
class mindspore.dataset.WikiTextDataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.WikiTextDataset](https://mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.WikiTextDataset.html#mindspore.dataset.WikiTextDataset).

## Differences

PyTorch: Read the WikiText2 dataset.

MindSpore: Read the WikiText2 dataset. Download dataset from web is not supported.
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
dataset = datasets.WikiText2(root, split=('train', 'valid', 'test'))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download WikiText2 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── wiki.train.tokens
#      ├── wiki.test.tokens
#      ├── wiki.valid.tokens
root = "/path/to/dataset_directory/"
ms_dataloader = ds.WikiTextDataset(root, usage='all')
```
