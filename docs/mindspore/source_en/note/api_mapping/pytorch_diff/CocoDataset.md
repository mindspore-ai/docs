# Differences with torch.torchvision.datasets.CocoDetection

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CocoDataset.md)

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

For more information, see [torchvision.datasets.CocoDetection](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.CocoDetection).

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

For more information, see [mindspore.dataset.CocoDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.CocoDataset.html#mindspore.dataset.CocoDataset).

## Differences

PyTorch: Input the COCO dataset, and return the created dataset object, which can be traversed to obtain data.

MindSpore: Input the COCO dataset and a specified task type (target detection, panorama segmentation, etc.), and return a dataset object with the given task type, which can be obtained by creating an iterator.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | annFile      | annotation_file    |- |
|     | Parameter3 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter4 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter5 | transforms    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter6 | -      | task  | Set the task type for reading COCO data |
|     | Parameter7 | -    | num_samples |  The number of images to be included in the dataset |
|     | Parameter8 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter9 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter10 | -   | decode | Whether to decode the images after reading |
|     | Parameter11 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter12 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter13 | -    | shard_id | The shard ID within num_shards |
|     | Parameter14 | -    | cache | Use tensor caching service to speed up dataset processing |
|     | Parameter15 | -    | extra_metadata | Flag to add extra meta-data to row |
|     | Parameter16 | -    | decrypt | Image decryption function |

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
