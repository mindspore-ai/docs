# Differences with torchaudio.transforms.GriffinLim

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GriffinLim.md)

## torchaudio.transforms.GriffinLim

```python
class torchaudio.transforms.GriffinLim(n_fft: int = 400, n_iter: int = 32, win_length: Optional[int] = None, hop_length: Optional[int] = None,
                                       window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>, power: float = 2.0,
                                       normalized: bool = False, wkwargs: Optional[dict] = None, momentum: float = 0.99,
                                       length: Optional[int] = None, rand_init: bool = True)
```

For more information, see [torchaudio.transforms.GriffinLim](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.GriffinLim.html).

## mindspore.dataset.audio.GriffinLim

```python
class mindspore.dataset.audio.GriffinLim(n_fft=400, n_iter=32, win_length=None, hop_length=None,
                                         window_type=WindowType.HANN, power=2.0,
                                         momentum=0.99, length=None, rand_init=True)
```

For more information, see [mindspore.dataset.audio.GriffinLim](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.GriffinLim.html#mindspore.dataset.audio.GriffinLim).

## Differences

PyTorch: Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation. Customized window function and different parameter configs for window function are both supported.

MindSpore: Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | n_fft     | n_fft     | - |
|     | Parameter2 | n_iter    | n_iter    | - |
|     | Parameter3 | win_length  | win_length    | - |
|     | Parameter4 | hop_length  | hop_length    | - |
|     | Parameter5 | window_fn   | window_type   | MindSpore only supports 5 window functions |
|     | Parameter6 | power  | power    | - |
|     | Parameter7 | normalized  | -    | Whether to normalize by magnitude after stft, not supported by MindSpore |
|     | Parameter8 | wkwargs   | -     | Arguments for window function, not supported by MindSpore |
|     | Parameter9 | momentum   | momentum     | - |
|     | Parameter10 | length   | length     | - |
|     | Parameter11 | rand_init  | rand_init     | - |

## Code Example

```python
import numpy as np

fake_input = np.ones((151, 36)).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T
torch.manual_seed(1)

transformer = T.GriffinLim(n_fft=300, n_iter=10, win_length=None, hop_length=None, window_fn=torch.hann_window, power=2, momentum=0.5)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([-0.0800,  0.1134, -0.0888,  ..., -0.0610, -0.0206, -0.1800])

# MindSpore
import mindspore as ms
import mindspore.dataset.audio as audio
ms.dataset.config.set_seed(3)

transformer = audio.GriffinLim(n_fft=300, n_iter=10, win_length=None, hop_length=None, window_type=audio.WindowType.HANN, power=2, momentum=0.5)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [-0.08666667  0.06763329 -0.03155987 ... -0.07218403 -0.01178891 -0.00664348]
```
