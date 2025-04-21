# Differences with torchvision.ops.nms

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/nms.md)

## torchvision.ops.nms

```python
torchvision.ops.nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float)
```

For more information, see [torchvision.ops.nms](https://pytorch.org/vision/0.9/ops.html#torchvision.ops.nms).

## mindspore.ops.NMSWithMask

```python
class mindspore.ops.NMSWithMask(iou_threshold=0.5)(bboxes)
```

For more information, see [mindspore.ops.NMSWithMask](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.NMSWithMask.html).

## Differences

PyTorch: Performs non-maximum suppression (NMS), shapes of `boxes` and `scores` are (N, 4) and (N, 1), represents the boxes and scores respectively.

MindSpore: Performs non-maximum suppression (NMS), shapes of `bboxes` is (N, 5), represents the boxes and scores in (x0, y0, x1, y1, score) format. Only supports up to 2864 input boxes at one time on Ascend.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | boxes   | -  | Bounding boxes, defined in the input list of MindSpore |
|     | Parameter2 | scores  | -  | Scores of bounding box, defined in the input list of MindSpore |
|     | Parameter3 | iou_threshold | iou_threshold  | Specify the threshold of overlap boxes with respect to IOU |
|Input | Input1 | -   | bboxes    | Bounding boxes with scores |
|Output | Output1 | indices | -  |  Indices of the elements that have been kept by NMS |
|     | Output2 | - | output_boxes  | A sorted list of bounding boxes by sorting the input bboxes in descending order of score |
|     | Output3 | - | output_idx | The indexes list `output_boxes` |
|     | Output4 | - | selected_mask | A mask list of valid output bounding boxes. True for keep, False for drop |

## Code Example

```python
# PyTorch
import torch
import torchvision as tv
import numpy as np

boxes = np.array([
    [0, 0, 4, 4],
    [0, 0, 3, 3],
    [0, 0, 2, 2],
    [0, 0, 1, 1]
]).astype(np.float32)

scores = np.array([0.8, 0.7, 0.6, 0.5]).astype(np.float32)

iou_threshold = 0.4

boxes_t = torch.from_numpy(boxes)
scores_t = torch.from_numpy(scores)

remain_boxes = tv.ops.nms(boxes_t, scores_t, iou_threshold)
print(remain_boxes)
# Out: tensor([0, 2, 3])

# MindSpore
import mindspore as ms
from mindspore import ops

box_with_score = np.column_stack((boxes, scores))
box_with_score_m = ms.Tensor(box_with_score)

output_boxes, output_idx, selected_mask = ops.NMSWithMask(iou_threshold)(box_with_score_m)
print(selected_mask)
# Out: [True False True True]
```