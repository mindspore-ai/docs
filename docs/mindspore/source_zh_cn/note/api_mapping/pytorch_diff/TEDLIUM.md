# 比较与torchaudio.datasets.TEDLIUM的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TEDLIUM.md)

## torchaudio.datasets.TEDLIUM

```python
class torchaudio.datasets.TEDLIUM(
    root: str,
    release: str = 'release1',
    subset: str = None,
    download: bool = False,
    audio_ext: str = '.sph')
```

更多内容详见[torchaudio.datasets.TEDLIUM](https://pytorch.org/audio/0.8.0/datasets.html#tedlium)。

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

更多内容详见[mindspore.dataset.TedliumDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.TedliumDataset.html#mindspore.dataset.TedliumDataset)。

## 差异对比

PyTorch：读取Tedlium数据集。

MindSpore：读取Tedlium数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | release      | release    |- |
|     | 参数3 | subset      | usage    | - |
|     | 参数4 | download    | -   | MindSpore不支持 |
|     | 参数5 | audio_ext    | extensions | - |
|     | 参数6 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数7 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数8 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数9 | -    | sampler  | 指定采样器 |
|     | 参数10 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数11 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数12 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

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
