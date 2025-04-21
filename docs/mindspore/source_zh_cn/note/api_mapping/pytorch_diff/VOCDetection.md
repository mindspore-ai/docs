# 比较与torchvision.datasets.VOCDetection的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/VOCDetection.md)

## torchvision.datasets.VOCDetection

```python
class torchvision.datasets.VOCDetection(
    root: str,
    year: str='2012',
    image_set: str='train',
    download: bool=False,
    transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None,
    transforms: Optional[Callable]=None
    )
```

更多内容详见[torchvision.datasets.VOCDetection](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.VOCDetection)。

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

更多内容详见[mindspore.dataset.VOCDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.VOCDataset.html#mindspore.dataset.VOCDataset)。

## 差异对比

PyTorch：生成PASCAL VOC 目标检测格式数据集。

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
|     | 参数15 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数16 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数17 | -    | cache | 指定单节点数据缓存服务 |
|     | 参数18 | -    | extra_metadata | 用于指定是否额外输出一个数据列用于表示图片元信息 |
|     | 参数19 | -    | decrypt | 图像解密函数 |

## 代码示例

```python
import mindspore.dataset as ds
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# In MindSpore, the generated dataset with different task setting has different output columns.

voc_dataset_dir = "/path/to/voc_dataset_directory/"

# task = Detection, output columns: [image, dtype=uint8], [bbox, dtype=float32], [label, dtype=uint32], [difficult, dtype=uint32], [truncate, dtype=uint32].
dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
for item in dataset:
    print("item:", item[0])
    print(len(item[0]))
    break
# out:
# item: [255 216 255 ...   3 255 217]
# 147025

# In torch, the output will be result of transform, eg. RandomCrop
root = "/path/to/voc_dataset_directory2/"

dataset = datasets.VOCDetection(root, image_set='train', year='2012', transform=T.ToTensor())
dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)
for epoch in range(1):
    for i, (data, label) in enumerate(dataloader):
        print((data, label)[0])

# out:
# tensor([[[[0.7176, 0.7176, 0.7216,  ..., 0.7843, 0.7843, 0.7843],
#           [0.7216, 0.7216, 0.7216,  ..., 0.7882, 0.7882, 0.7882],
#           [0.7216, 0.7255, 0.7255,  ..., 0.7882, 0.7882, 0.7882],
#           ...,
#           ...
#          ...,
#           [0.6667, 0.6667, 0.6667,  ..., 0.8118, 0.8118, 0.8078],
#           [0.6627, 0.6627, 0.6588,  ..., 0.8078, 0.8039, 0.8000],
#           [0.6627, 0.6627, 0.6588,  ..., 0.8078, 0.8039, 0.8000]]]])
#  {'annotation': {'folder': ['VOC2012'], 'filename': ['61.jpg'], 'source': {'database': ['simulate VOC2007 Database'],
#  'annotation': ['simulate VOC2007'], 'image': ['flickr']}, 'size': {'width': ['500'], 'height': ['333'], 'depth': ['3']}, 'segmented': ['1'],
#  'object': [{'name': ['train'], 'pose': ['Unspecified'], 'truncated': ['0'], 'difficult': ['0'], 'bndbox': {'xmin': ['252'], 'ymin': ['42'],
#  'xmax': ['445'], 'ymax': ['282']}}, {'name': ['person'], 'pose': ['Frontal'], 'truncated': ['0'], 'difficult': ['0'], 'bndbox': {'xmin': ['204'],
#  'ymin': ['198'], 'xmax': ['271'], 'ymax': ['293']}}]}}
```
