# 比较与torchvision.datasets.CocoDetection的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CocoDataset.md)

## torchvision.datasets.CocoDetection

```python
class torchvision.datasets.CocoDetection(
    root: str,
    annFile: str,
    transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None,
    transforms: Optional[Callable]=None
    )
```

更多内容详见[torchvision.datasets.CocoDetection](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.CocoDetection)。

## mindspore.dataset.CocoDataset

```python
class mindspore.dataset.CocoDataset(
    dataset_dir,
    annotation_file,
    task="Detection",
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

更多内容详见[mindspore.dataset.CocoDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.CocoDataset.html#mindspore.dataset.CocoDataset)。

## 差异对比

PyTorch：输入COCO格式数据集，返回创建出的数据集对象，可通过遍历数据集对象获取数据。

MindSpore：输入COCO格式数据集及指定任务类型（目标检测，全景分割等），返回给定任务类型的数据集对象，可通过创建迭代器获取数据。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | annFile      | annotation_file    |- |
|     | 参数3 | transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数4 | target_transform    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数5 | transforms    | -   | MindSpore通过 `mindspore.dataset.map` 操作支持 |
|     | 参数6 | -      | task  | 指定COCO数据的任务类型 |
|     | 参数7 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数8 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数9 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数10 | -    | decode | 指定是否对图像进行解码 |
|     | 参数11 | -    | sampler  | 指定采样器 |
|     | 参数12 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数13 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数14 | -    | cache | 指定单节点数据缓存服务 |
|     | 参数15 | -    | extra_metadata | 用于指定是否额外输出一个数据列用于表示图片元信息 |
|     | 参数16 | -    | decrypt | 图像解密函数 |

## 代码示例

```python
import mindspore.dataset as ds
import torchvision.datasets as datasets
import torchvision.transforms as T

# In MindSpore, CocoDataset supports four kinds of tasks, which are Object Detection, Keypoint Detection, Stuff Segmentation and Panoptic Segmentation of 2017 Train/Val/Test dataset.

coco_dataset_dir = "/path/to/coco_dataset_directory/images"
coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"

# Read COCO data for Detection task. Output columns: [image, dtype=uint8], [bbox, dtype=float32], [category_id, dtype=uint32], [iscrowd, dtype=uint32]
dataset = ds.CocoDataset(
    dataset_dir=coco_dataset_dir,
    annotation_file=coco_annotation_file,
    task='Detection',
    decode=True,
    shuffle=False,
    extra_metadata=True)

dataset = dataset.rename("_meta-filename", "filename")
file_name = []
bbox = []
category_id = []
iscrowd = []
for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    file_name.append(data["filename"])
    bbox.append(data["bbox"])
    category_id.append(data["category_id"])
    iscrowd.append(data["iscrowd"])

print(file_name[0])
print(bbox[0])
print(category_id[0])
print(iscrowd[0])
# out:
# 000000391895
# [[10. 10. 10. 10.]
# [70. 70. 70. 70.]]
# [[1]
# [7]]
# [[0]
# [0]]

# In torch, the output will be result of transform, eg. Tensor
root = "/path/to/coco_dataset_directory/images"
annFile = "/path/to/coco_dataset_directory/annotation_file"

# Convert a PIL Image or numpy.ndarray to tensor.
dataset = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
for item in dataset:
    print("item:", item[0])
    break
# out:
# loading annotations into memory...
# Done (t=0.00s)
# creating index...
# index created!
# item: tensor([[[0.8588, 0.8549, 0.8549,  ..., 0.7529, 0.7529, 0.7529,
#        [0.8549, 0.8549, 0.8510,  ..., 0.7529, 0.7529, 0.7529],
#        [0.8549, 0.8510, 0.8510,  ..., 0.7529, 0.7529, 0.7529],
#        ...,
#
#        ...,
#        [0.8471, 0.8510, 0.8549,  ..., 0.7412, 0.7333, 0.7294],
#        [0.8549, 0.8549, 0.8549,  ..., 0.7412, 0.7333, 0.7294],
#        [0.8627, 0.8627, 0.8549,  ..., 0.7412, 0.7333, 0.7294]]])
```
