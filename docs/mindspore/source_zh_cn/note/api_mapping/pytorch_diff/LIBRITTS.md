# 比较与torchaudio.datasets.LIBRITTS的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LIBRITTS.md)

## torchaudio.datasets.LIBRITTS

```python
class torchaudio.datasets.LIBRITTS(
    root: str,
    url: str = 'train-clean-100',
    folder_in_archive: str = 'LibriTTS',
    download: bool = False)
```

更多内容详见[torchaudio.datasets.LIBRITTS](https://pytorch.org/audio/0.8.0/datasets.html#libritts)。

## mindspore.dataset.LibriTTSDataset

```python
class mindspore.dataset.LibriTTSDataset(
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

更多内容详见[mindspore.dataset.LibriTTSDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.LibriTTSDataset.html#mindspore.dataset.LibriTTSDataset)。

## 差异对比

PyTorch：读取LibriTTS数据集。

MindSpore：读取LibriTTS数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | url      | usage    |- |
|     | 参数3 | folder_in_archive      | -    | MindSpore不支持 |
|     | 参数4 | download    | -   | MindSpore不支持 |
|     | 参数5 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数6 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数7 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数8 | -    | sampler  | 指定采样器 |
|     | 参数9 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数10 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数11 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# PyTorch
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.LIBRITTS(root, url='train-clean-100')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download LibriTTS dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── dev-clean
#      │    ├── 116
#      │    │    ├── 288045
#      |    |    |    ├── 116_288045.trans.tsv
#      │    │    │    ├── 116_288045_000003_000000.wav
#      │    │    │    └──...
#      │    │    ├── 288046
#      |    |    |    ├── 116_288046.trans.tsv
#      |    |    |    ├── 116_288046_000003_000000.wav
#      │    |    |    └── ...
#      |    |    └── ...
#      │    ├── 1255
#      │    │    ├── 138279
#      |    |    |    ├── 1255_138279.trans.tsv
#      │    │    │    ├── 1255_138279_000001_000000.wav
#      │    │    │    └── ...
#      │    │    ├── 74899
#      |    |    |    ├── 1255_74899.trans.tsv
#      |    |    |    ├── 1255_74899_000001_000000.wav
#      │    |    |    └── ...
#      |    |    └── ...
#      |    └── ...
#      └── ...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.LibriTTSDataset(root, usage='train-clean-100')
```
