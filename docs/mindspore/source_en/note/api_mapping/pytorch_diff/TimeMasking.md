# Differences with torchaudio.transforms.TimeMasking

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TimeMasking.md)

## torchaudio.transforms.TimeMasking

```python
class torchaudio.transforms.TimeMasking(time_mask_param: int, iid_masks: bool = False)
```

For more information, see [torchaudio.transforms.TimeMasking](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.TimeMasking.html).

## mindspore.dataset.audio.TimeMasking

```python
class mindspore.dataset.audio.TimeMasking(iid_masks=False, time_mask_param=0, mask_start=0, mask_value=0.0)
```

For more information, see [mindspore.dataset.audio.TimeMasking](https://mindspore.cn/docs/en/br_base/api_python/dataset_audio/mindspore.dataset.audio.TimeMasking.html#mindspore.dataset.audio.TimeMasking).

## Differences

PyTorch: Apply masking to a spectrogram in the time domain.

MindSpore: Apply masking to a spectrogram in the time domain. Variable `mask_value` is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | time_mask_param    | time_mask_param    | - |
|     | Parameter2 | iid_masks   | iid_masks   | - |
|     | Parameter3 | -   | mask_start   | Starting point to apply mask  |
|     | Parameter4 | -   | mask_value   | Value to assign to the masked location, can not be changed during computing in MindSpore  |

## Code Example

```python
import numpy as np

fake_wav = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                      [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T
torch.manual_seed(1)

transformer = T.TimeMasking(time_mask_param=2, iid_masks=True)
torch_result = transformer(torch.from_numpy(fake_wav), mask_value=0.0)
print(torch_result)
# Out: tensor([[[ 0.0000,  0.8517,  0.0716, -0.4544],
#               [ 0.0000,  0.3353,  1.7414,  0.1231]]])

# MindSpore
import mindspore as ms
import mindspore.dataset.audio as audio
ms.dataset.config.set_seed(2)

transformer = audio.TimeMasking(time_mask_param=2, iid_masks=True, mask_start=0, mask_value=0.0)
ms_result = transformer(fake_wav)
print(ms_result)
# Out: [[[ 0.          0.85174704  0.07162686 -0.45436913]
#        [ 0.          0.33526883  1.7413973   0.12313101]]]
```
