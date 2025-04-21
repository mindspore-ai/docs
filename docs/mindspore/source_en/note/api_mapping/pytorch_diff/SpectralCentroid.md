# Differences with torchaudio.transforms.SpectralCentroid

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SpectralCentroid.md)

## torchaudio.transforms.SpectralCentroid

```python
class torchaudio.transforms.SpectralCentroid(sample_rate: int, n_fft: int = 400, win_length: Optional[int] = None,
                                             hop_length: Optional[int] = None, pad: int = 0,
                                             window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
                                             wkwargs: Optional[dict] = None)
```

For more information, see [torchaudio.transforms.SpectralCentroid](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.SpectralCentroid.html).

## mindspore.dataset.audio.SpectralCentroid

```python
class mindspore.dataset.audio.SpectralCentroid(sample_rate, n_fft=400, win_length=None, hop_length=None,
                                               pad=0, window=WindowType.HANN)
```

For more information, see [mindspore.dataset.audio.SpectralCentroid](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.SpectralCentroid.html#mindspore.dataset.audio.SpectralCentroid).

## Differences

PyTorch: Compute the spectral centroid for each channel along the time axis. Customized window function and different parameter configs for window function are both supported.

MindSpore: Compute the spectral centroid for each channel along the time axis.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | sample_rate      | sample_rate      | - |
|     | Parameter2 | n_fft    |  n_fft  | - |
|     | Parameter3 | win_length  | win_length    | - |
|     | Parameter4 | hop_length  | hop_length    | - |
|     | Parameter5 | pad    | pad   |  |
|     | Parameter6 | window_fn   | window     | MindSpore only supports 5 window functions |
|     | Parameter7 | wkwargs  | -    | Arguments for window function, not supported by MindSpore |

## Code Example

```python
import numpy as np

fake_input = np.array([[[1, 1, 2, 2, 3, 3, 4]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.SpectralCentroid(sample_rate=44100, n_fft=8, window_fn=torch.hann_window)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[4436.1182, 3768.7986]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.SpectralCentroid(sample_rate=44100, n_fft=8, window=audio.WindowType.HANN)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[4436.117  3768.7979]]]]
```
