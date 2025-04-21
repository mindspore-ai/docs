# 比较与torchvision.ops.nms的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/nms.md)

## torchvision.ops.nms

```python
torchvision.ops.nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float)
```

更多内容详见[torchvision.ops.nms](https://pytorch.org/vision/0.9/ops.html#torchvision.ops.nms)。

## mindspore.ops.NMSWithMask

```python
class mindspore.ops.NMSWithMask(iou_threshold=0.5)(bboxes)
```

更多内容详见[mindspore.ops.NMSWithMask](https://mindspore.cn/docs/zh-CN/br_base/api_python/ops/mindspore.ops.NMSWithMask.html)。

## 差异对比

PyTorch：非极大值抑制算法。参数 `boxes` 与 `scores` 维度分别为(N, 4)和(N, 1)，表示bbox与score。

MindSpore：非极大值抑制算法，参数 `bboxes` 维度为(N, 5)，以(x0、y0、x1、y1, score)形式表示框与分数。Ascend平台最大支持2864个输入框。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | boxes   | -  | 边界框，此参数位于MindSpore算子的输入参数列表中 |
|     | 参数2 | scores  | -  | 边界框的分数，此参数位于MindSpore算子的输入参数列表中 |
|     | 参数3 | iou_threshold | iou_threshold  | 指定删除框的IOU的阈值 |
|输入 | 输入1 | -   | bboxes    | 边界框与分数 |
|输出 | 输出1 | indices | -  | PyTorch表示为NMS后的bbox索引 |
|     | 输出2 | - | output_boxes  | 按score排序后的bbox列表 |
|     | 输出3 | - | output_idx | `output_boxes` 的索引列表 |
|     | 输出4 | - | selected_mask | 表示NMS后的box掩码，True为剩余bbox，False为丢弃的bbox |

## 代码示例

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
