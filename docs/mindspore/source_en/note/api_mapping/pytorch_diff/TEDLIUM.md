# Differences with torchaudio.datasets.TEDLIUM

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TEDLIUM.md)

## torchaudio.datasets.TEDLIUM

```python
class torchaudio.datasets.TEDLIUM(
    root: str,
    release: str = 'release1',
    subset: str = None,
    download: bool = False,
    audio_ext: str = '.sph')
```

For more information, see [torchaudio.datasets.TEDLIUM](https://pytorch.org/audio/0.8.0/datasets.html#tedlium).

## mindspore.dataset.TedliumDataset

```python
class mindspore.dataset.TedliumDataset(
    dataset_dir,
    release,
    usage=None,
    extensions=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.TedliumDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.TedliumDataset.html#mindspore.dataset.TedliumDataset).

## Differences

PyTorch: Read the Tedlium dataset.

MindSpore: Read the Tedlium dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | release      | release    |- |
|     | Parameter3 | subset      | usage    |- |
|     | Parameter4 | download    | -   | Not supported by MindSpore |
|     | Parameter5 | audio_ext      | extensions    |- |
|     | Parameter6 | -    | num_samples |  The number of images to be included in the dataset |
|     | Parameter7 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter8 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter9 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter10 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter11 | -    | shard_id | The shard ID within num_shards |
|     | Parameter12 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.TEDLIUM(root, release='release1')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download Tedlium dataset files, unzip into the following structure
# .
# └──TEDLIUM_release1
#     └── dev
#         ├── sph
#             ├── AlGore_2009.sph
#             ├── BarrySchwartz_2005G.sph
#         ├── stm
#             ├── AlGore_2009.stm
#             ├── BarrySchwartz_2005G.stm
#     └── test
#         ├── sph
#             ├── AimeeMullins_2009P.sph
#             ├── BillGates_2010.sph
#         ├── stm
#             ├── AimeeMullins_2009P.stm
#             ├── BillGates_2010.stm
#     └── train
#         ├── sph
#             ├── AaronHuey_2010X.sph
#             ├── AdamGrosser_2007.sph
#         ├── stm
#             ├── AaronHuey_2010X.stm
#             ├── AdamGrosser_2007.stm
#     └── readme
#     └── TEDLIUM.150k.dic
root = "/path/to/dataset_directory/"
ms_dataloader = ds.TedliumDataset(root, release='release1')
```
