# 比较与torchaudio.datasets.YESNO的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/YESNO.md)

## torchaudio.datasets.YESNO

```python
class torchaudio.datasets.YESNO(
    root: str,
    url: str = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz',
    folder_in_archive: str = 'waves_yesno',
    download: bool = False)
```

更多内容详见[torchaudio.datasets.YESNO](https://pytorch.org/audio/0.8.0/datasets.html#yesno)。

## mindspore.dataset.YesNoDataset

```python
class mindspore.dataset.YesNoDataset(
    dataset_dir,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.YesNoDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.YesNoDataset.html#mindspore.dataset.YesNoDataset)。

## 差异对比

PyTorch：读取YesNo数据集。

MindSpore：读取YesNo数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | url      | -    |MindSpore不支持 |
|     | 参数3 | folder_in_archive      | - | MindSpore不支持  |
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
dataset = datasets.YESNO(root, url='http://www.openslr.org/resources/1/waves_yesno.tar.gz')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download YesNo dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── 1_1_0_0_1_1_0_0.wav
#      ├── 1_0_0_0_1_1_0_0.wav
#      ├── 1_1_0_0_1_1_0_0.wav
#      └──....
root = "/path/to/dataset_directory/"
ms_dataloader = ds.YesNoDataset(root)
```
