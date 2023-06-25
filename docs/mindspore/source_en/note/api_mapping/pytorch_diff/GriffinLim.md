# Differences with torchaudio.transforms.GriffinLim

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GriffinLim.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.dataset.audio.GriffinLim](https://mindspore.cn/docs/en/master/api_python/dataset_audio/mindspore.dataset.audio.GriffinLim.html#mindspore.dataset.audio.GriffinLim).

## Differences

PyTorch：Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation. Customized window function and different parameter configs for window function are both supported.

MindSpore：Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | n_fft     | n_fft     | - |
|     | Parameter2 | n_iter    | n_iter    | - |
|     | Parameter3 | win_length  | win_length    | - |
|     | Parameter4 | hop_length  | hop_length    | - |
|     | Parameter5 | window_fn   | window_type   | MindSpore only support 5 window functions |
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

transformer = T.GriffinLim(n_fft=300, n_iter=10, win_length=None, hop_length=None, window_fn=torch.hann_window, power=2, momentum=0.5)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([-0.1000, -0.0116,  0.0687,  ..., -0.0897, -0.0263,  0.0067])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.GriffinLim(n_fft=300, n_iter=10, win_length=None, hop_length=None, window_type=audio.WindowType.HANN, power=2, momentum=0.5)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [ 0.06666667  0.06637156  0.08702356 ...  0.04688901 -0.05855678 0.03994351]
```
