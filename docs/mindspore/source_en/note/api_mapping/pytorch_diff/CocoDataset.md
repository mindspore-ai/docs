# Function differences with torch.torchvision.datasets.CocoDetection

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CocoDataset.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torchvision.datasets.CocoDetection](https://pytorch.org/vision/0.10/datasets.html#torchvision.datasets.CocoDetection).

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
    extra_metadata=False
    )
```

For more information, see [mindspore.dataset.CocoDataset](https://mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.CocoDataset.html#mindspore.dataset.CocoDataset).

## Differences

PyTorch: Enter the COCO dataset, return the transformed version with standard interfaces.

MindSpore: Enter the COCO dataset, return the transformed version for four kinds of tasks with standard interfaces.

## Code Example

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
    file_name.append(text.to_str(data["filename"]))
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
