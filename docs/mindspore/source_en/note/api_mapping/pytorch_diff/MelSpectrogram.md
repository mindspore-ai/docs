# Differences with torchaudio.transforms.MelSpectrogram

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MelSpectrogram.md)

## torchaudio.transforms.MelSpectrogram

```python
class torchaudio.transforms.MelSpectrogram(sample_rate: int = 16000, n_fft: int = 400, win_length: Optional[int] = None,
                                           hop_length: Optional[int] = None, f_min: float = 0.0, f_max: Optional[float] = None,
                                           pad: int = 0, n_mels: int = 128, window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
                                           power: Optional[float] = 2.0, normalized: bool = False, wkwargs: Optional[dict] = None,
                                           center: bool = True, pad_mode: str = 'reflect', onesided: bool = True, norm: Optional[str] = None)
```

For more information, see [torchaudio.transforms.MelSpectrogram](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.MelSpectrogram.html).

## mindspore.dataset.audio.MelSpectrogram

```python
class mindspore.dataset.audio.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=None,
                                             hop_length=None, f_min=0.0, f_max=None,
                                             pad=0, n_mels=128, window=WindowType.HANN, power=2.0, normalized=False,
                                             center=True, pad_mode=BorderType.REFLECT, onesided=True, norm=NormType.NONE, mel_scale=MelType.HTK)
```

For more information, see [mindspore.dataset.audio.MelSpectrogram](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.MelSpectrogram.html#mindspore.dataset.audio.MelSpectrogram).

## Differences

PyTorch: Create MelSpectrogram for a raw audio signal. Customized window function and different parameter configs for window function are both supported.

MindSpore: Create MelSpectrogram for a raw audio signal.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | sample_rate     | sample_rate      | - |
|     | Parameter2 | win_length     | win_length    | - |
|     | Parameter3 | hop_length   | hop_length    | - |
|     | Parameter4 | n_fft   | n_fft    | - |
|     | Parameter5 | f_min    | f_min   | - |
|     | Parameter6 | f_max   | f_max    | - |
|     | Parameter7 | pad   | pad  | - |
|     | Parameter8 | n_mels    | n_mels     | - |
|     | Parameter9 | window_fn    | window      | MindSpore only supports 5 window functions |
|     | Parameter10 | power    | power     | - |
|     | Parameter11 | normalized   | normalized     | - |
|     | Parameter12 | wkwargs    | -     | Arguments for window function, not supported by MindSpore |
|     | Parameter13 | center   | center     | - |
|     | Parameter14 | pad_mode    | pad_mode     | - |
|     | Parameter15 | onesided    | onesided     | - |
|     | Parameter16 | norm    | norm     | - |
|     | Parameter17 | -    | mel_scale      | Mel scale to use |

## Code Example

```python
import numpy as np

fake_input = np.array([[[1, 1, 2, 2, 3, 3, 4]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.MelSpectrogram(sample_rate=16000, n_fft=4, win_length=2, hop_length=4, window_fn=torch.hann_window)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[0.0000, 0.0000],
#                ...
#                [0.5235, 4.7117],
#                [0.4765, 4.2883],
#                ...
#                [0.0000, 0.0000]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.MelSpectrogram(sample_rate=16000, n_fft=4, win_length=2, hop_length=4, window=audio.WindowType.HANN)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[0.         0.        ]
#         ...
#         [0.52353615 4.7118254 ]
#         [0.47646385 4.2881746 ]
#         ...
#         [0.         0.        ]]]]
```
