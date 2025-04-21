# 比较与torchvision.datasets.VOCSegmentation的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/VOCSegmentation.md)

## torchvision.datasets.VOCSegmentation

```python
class torchvision.datasets.VOCSegmentation(
    root: str,
    year: str='2012',
    image_set: str='train',
    download: bool=False,
    transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None,
    transforms: Optional[Callable]=None
    )
```

更多内容详见[torchvision.datasets.VOCSegmentation](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.VOCSegmentation)。

## mindspore.dataset.VOCDataset

```python
class mindspore.dataset.VOCDataset(
    dataset_dir,
    task="Segmentation",
    usage="train",
    class_indexing=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    decode=False,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None,
    extra_metadata=False,
    decrypt=None
    )
```

更多内容详见[mindspore.dataset.VOCDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.VOCDataset.html#mindspore.dataset.VOCDataset)。

## 差异对比

PyTorch：生成PASCAL VOC图像分割格式数据集。

MindSpore：用于读取和分析VOC数据集的源数据集。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | year      | -    | MindSpore不支持 |
|     | 参数3 | image_set      | usage  |- |
|     | 参数4 | download      | -    | MindSpore不支持 |
|     | 参数5 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数6 | target_transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数7 | transforms    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数8 | -      | task  | 指定读取VOC数据的任务类型 |
|     | 参数9 | -      | class_indexing  | 指定一个从label名称到label索引的映射 |
|     | 参数10 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数11 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数12 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数13 | -    | decode | 指定是否对图像进行解码 |
|     | 参数14 | -    | sampler  | 指定采样器 |
|     | 参数15 | -    | num_shards | 指定分布式训练时，将数据集进行划分的分片数 |
|     | 参数16 | -    | shard_id | 指定分布式训练时，使用的分片ID号 |
|     | 参数17 | -    | cache | 指定单节点数据缓存服务 |
|     | 参数18 | -    | extra_metadata | 指定是否额外输出一个表示图片元信息的数据列 |
|     | 参数19 | -    | decrypt | 图像解密函数 |

## 代码示例

```python
import mindspore.dataset as ds
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# In MindSpore, the generated dataset with different task setting has different output columns.

voc_dataset_dir = "/path/to/voc_dataset_directory/"

# task = Segmentation, output columns: [image, dtype=uint8], [target,dtype=uint8].
dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir,
                                                    task="Segmentation",
                                                    usage="train")
for item in dataset:
    print("item:", item[0])
    print(len(item[0]))
    break
# Out:
# item: [255 216 255 ...  73 255 217]
# 52544

# In torch, the output will be result of transform, eg. RandomCrop
root = "/path/to/voc_dataset_directory2/"

dataset = datasets.VOCSegmentation(root, image_set='train', year='2012', transform=T.RandomCrop(300))
print(dataset)
print(type(dataset))
```
