# Differences with torchtext.datasets.IWSLT2017

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/IWSLT2017.md)

## torchtext.datasets.IWSLT2017

```python
class torchtext.datasets.IWSLT2017(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'),
    language_pair: Sequence = ('de', 'en'))
```

For more information, see [torchtext.datasets.IWSLT2017](https://pytorch.org/text/0.9.0/datasets.html#iwslt2017).

## mindspore.dataset.IWSLT2017Dataset

```python
class mindspore.dataset.IWSLT2017Dataset(
    dataset_dir,
    usage=None,
    language_pair=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.IWSLT2017Dataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.IWSLT2017Dataset.html#mindspore.dataset.IWSLT2017Dataset).

## Differences

PyTorch: Read the IWSLT2017 dataset.

MindSpore: Read the IWSLT2017 dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    |- |
|     | Parameter3 | language_pair      | language_pair    |- |
|     | Parameter4 | -    | num_samples | The number of images to be included in the dataset |
|     | Parameter5 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter6 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter7 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter8 | -    | shard_id | The shard ID within num_shards |
|     | Parameter9 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchtext.datasets as datasets

root = "/path/to/dataset_root/"
train_iter, valid_iter, test_iter = datasets.IWSLT2017(root, split=('train', 'valid', 'test'))
data = next(iter(train_iter))

# MindSpore
import mindspore.dataset as ds

# Download IWSLT2017 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      └── DeEnItNlRo
#          └── DeEnItNlRo
#              └── DeEnItNlRo-DeEnItNlRo
#                  ├── IWSLT17.TED.dev2010.de-en.de.xml
#                  ├── train.tags.de-en.de
#                  ├── ...
root = "/path/to/dataset_directory/"
dataset = ds.IWSLT2017Dataset(root, usage='all')
data = next(iter(dataset))
```
