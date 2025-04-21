# 比较与torchvision.ops.roi_align的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/roi_align.md)

## torchvision.ops.roi_align

```python
torchvision.ops.roi_align(input: torch.Tensor, boxes: torch.Tensor, output_size: None, spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False)
```

更多内容详见[torchvision.ops.roi_align](https://pytorch.org/vision/0.9/ops.html#torchvision.ops.roi_align.html)。

## mindspore.ops.ROIAlign

```python
class mindspore.ops.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1)(features, rois)
```

更多内容详见[mindspore.ops.ROIAlign](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/ops/mindspore.ops.ROIAlign.html)。

## 差异对比

PyTorch：感兴趣区域对齐（RoI Align）。

MindSpore：感兴趣区域对齐（RoI Align）。与PyTorch相比，参数输入形式不同，对齐模式的实现也不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | input   | -  | 输入特征，此参数位于MindSpore算子的输入参数列表中 |
|     | 参数2 | boxes  | -  | 边界框坐标，此参数位于MindSpore算子的输入参数列表中 |
|     | 参数3 | output_size | [pooled_height, pooled_width]  | 特征尺寸，MindSopre分别用2个参数表示 |
|     | 参数4 | spatial_scale  | spatial_scale  | - |
|     | 参数5 | sampling_ratio | sample_num  | - |
|     | 参数6 | aligned | roi_end_mode  | 对齐的模式。PyTorch对应的参数值为False和True，MindSpore对应的参数值为0和1 |
|输入 | 输入1 | -   | features    | 输入特征 |
|     | 输入2 | -   | rois    | roi坐标 |
|输出 | 输出1 | Tensor | Tensor  | - |

## 代码示例

```python
# PyTorch
import numpy as np
import torch
import torchvision as tv

pooled_height, pooled_width, spatial_scale, sample_num, roi_end_mode = 3, 3, 0.25, 2, 1

features = np.array([[[[1., 2.], [3., 4.]]]]).astype(np.float32)
rois = np.array([[0, 0.2, 0.3, 0.2, 0.3]]).astype(np.float32)

features_t = torch.from_numpy(features)
rois_t = torch.from_numpy(rois)

output = tv.ops.roi_align(features_t, rois_t, (pooled_height, pooled_width), spatial_scale, sample_num, 0)
print(output)
# Out: tensor([[[[1.7000, 2.0333, 2.3667],
#                [2.3667, 2.7000, 3.0333],
#                [3.0333, 3.3667, 3.7000]]]])

# MindSpore
import mindspore as ms
from mindspore import ops

features = ms.Tensor(np.array([[[[1., 2.], [3., 4.]]]]), ms.float32)
rois = ms.Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), ms.float32)

roi_align = ops.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num, 0)

output = roi_align(features, rois)
print(output)
# Out: [[[[1.7       2.0333333 2.3666668]
#         [2.3666668 2.7       3.0333335]
#         [3.0333333 3.3666668 3.7      ]]]]
```
