# Differences with torchaudio.transforms.Resample

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Resample.md)

## torchaudio.transforms.Resample

```python
class torchaudio.transforms.Resample(orig_freq: int = 16000, new_freq: int = 16000, resampling_method: str = 'sinc_interpolation')
```

For more information, see [torchaudio.transforms.Resample](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.Resample.html).

## mindspore.dataset.audio.Resample

```python
class mindspore.dataset.audio.Resample(orig_freq=16000, new_freq=16000, resample_method=ResampleMethod.SINC_INTERPOLATION,
                                       lowpass_filter_width=6, rolloff=0.99, beta=None)
```

For more information, see [mindspore.dataset.audio.Resample](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.Resample.html#mindspore.dataset.audio.Resample).

## Differences

PyTorch: Resample a signal from one frequency to another.

MindSpore: Resample a signal from one frequency to another. Extra filter option is supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | orig_freq     | orig_freq    | - |
|     | Parameter2 | new_freq    | new_freq   | - |
|     | Parameter3 | resampling_method    | resample_method     | - |
|     | Parameter4 | -   | lowpass_filter_width    | Sharpness of the filter |
|     | Parameter5 | -   | rolloff     | The roll-off frequency of the filter |
|     | Parameter6 | -   | beta     | The shape parameter used for kaiser window |

## Code Example

```python
import numpy as np

fake_input = np.array([[[[-0.2197528, 0.3821656]]],
                      [[[0.57418776, 0.46741104]]],
                      [[[0.76986176, -0.5793846]]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.Resample(orig_freq=16000, new_freq=24000)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[-0.2140,  0.2226,  0.3510]]],
#              [[[ 0.5728,  0.6145,  0.2789]]],
#              [[[ 0.7568, -0.1601, -0.6101]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.Resample(orig_freq=16000, new_freq=24000)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[-0.21398525  0.22255361  0.35099414]]]
#       [[[ 0.5728122   0.614469    0.2788692 ]]]
#       [[[ 0.75675076 -0.16008556 -0.61005235]]]]
```
