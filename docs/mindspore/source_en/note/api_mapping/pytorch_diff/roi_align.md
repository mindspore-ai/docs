# Differences with torchvision.ops.roi_align

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/roi_align.md)

## torchvision.ops.roi_align

```python
torchvision.ops.roi_align(input: torch.Tensor, boxes: torch.Tensor, output_size: None, spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False)
```

For more information, see [torchvision.ops.roi_align](https://pytorch.org/vision/0.9/ops.html#torchvision.ops.roi_align).

## mindspore.ops.ROIAlign

```python
class mindspore.ops.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1)(features, rois)
```

For more information, see [mindspore.ops.ROIAlign](https://mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ROIAlign.html).

## Differences

PyTorch: Computes the Region of Interest (RoI) Align operator.

MindSpore: Computes the Region of Interest (RoI) Align operator. The input list is different and align mode is different.

| Categories | Subcategories |PyTorch | MindSpore | Difference || --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | input   | -  | The input features, defined in the input list of MindSpore  |
|     | Parameter2 | boxes  | -  | Box coordinates, defined in the input list of MindSpore |
|     | Parameter3 | output_size | [pooled_height, pooled_width]  | The size of output features, defined in two parameters in MindSpore |
|     | Parameter4 | spatial_scale  | spatial_scale  | - |
|     | Parameter5 | sampling_ratio | sample_num  | - |
|     | Parameter6 | aligned | roi_end_mode  | Align mode.  are False and True, while parameter values for MindSpore are 0 and 1. |
|Input | Input1 | -   | features    | The input features |
|     | Input2 | -   | rois    | The input box coordinates |
|Output | Output1 | Tensor | Tensor  |- |

## Code Example

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