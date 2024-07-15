# 比较与torchaudio.datasets.GTZAN的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GTZAN.md)

## torchaudio.datasets.GTZAN

```python
class torchaudio.datasets.GTZAN(
    root: str,
    url: str = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
    folder_in_archive: str = 'genres',
    download: bool = False,
    subset: str = None)
```

更多内容详见[torchaudio.datasets.GTZAN](https://pytorch.org/audio/0.8.0/datasets.html#gtzan)。

## mindspore.dataset.GTZANDataset

```python
class mindspore.dataset.GTZANDataset(
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

更多内容详见[mindspore.dataset.GTZANDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.GTZANDataset.html#mindspore.dataset.GTZANDataset)。

## 差异对比

PyTorch：读取GTZAN数据集。

MindSpore：读取GTZAN数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | url      | -   |MindSpore不支持 |
|     | 参数3 | folder_in_archive      | -    | MindSpore不支持 |
|     | 参数4 | download    | -   | MindSpore不支持 |
|     | 参数5 | subset      | usage    | - |
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
dataset = datasets.GTZAN(root, url='http://opihi.cs.uvic.ca/sound/genres.tar.gz')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download GTZAN dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── blues
#      │    ├──blues.00000.wav
#      │    ├──blues.00001.wav
#      │    ├──blues.00002.wav
#      │    ├──...
#      ├── disco
#      │    ├──disco.00000.wav
#      │    ├──disco.00001.wav
#      │    ├──disco.00002.wav
#      │    └──...
#      └──...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.GTZANDataset(root, usage='all')
```
