# Function differences with torchaudio.transforms.InverseMelScale

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/InverseMelScale.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchaudio.transforms.InverseMelScale

```python
class torchaudio.transforms.InverseMelScale(n_stft: int, n_mels: int = 128, sample_rate: int = 16000, f_min: float = 0.0, f_max: Optional[float] = None,
                                            max_iter: int = 100000, tolerance_loss: float = 1e-05, tolerance_change: float = 1e-08, sgdargs: Optional[dict] = None,
                                            norm: Optional[str] = None)
```

For more information, see [torchaudio.transforms.InverseMelScale](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.InverseMelScale.html).

## mindspore.dataset.audio.InverseMelScale

```python
class mindspore.dataset.audio.InverseMelScale(n_stft, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None,
                                              max_iter=100000, tolerance_loss=1e-5, tolerance_change=1e-8, sgdargs=None,
                                              norm=NormType.NONE, mel_type=MelType.HTK)
```

For more information, see [mindspore.dataset.audio.InverseMelScale](https://mindspore.cn/docs/en/master/api_python/dataset_audio/mindspore.dataset.audio.InverseMelScale.html#mindspore.dataset.audio.InverseMelScale).

## Differences

PyTorch: Solve for a normal STFT from a mel frequency STFT, using a conversion matrix.

MindSpore: Solve for a normal STFT from a mel frequency STFT, using a conversion matrix. Mel scale can be specified.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | n_stft     | n_stft     | - |
|     | Parameter2 | n_mels    | n_mels    | - |
|     | Parameter3 | sample_rate  | sample_rate  | - |
|     | Parameter4 | f_min  | f_min    | - |
|     | Parameter5 | f_max   | f_max     | - |
|     | Parameter6 | max_iter   | max_iter     | - |
|     | Parameter7 | tolerance_loss   | tolerance_loss     | - |
|     | Parameter8 | tolerance_change   | tolerance_change     | - |
|     | Parameter9 | sgdargs   | sgdargs     | - |
|     | Parameter10 | norm   | norm     | - |
|     | Parameter11 | -   | mel_type      | Mel scale to use |

## Code Example

```python
import numpy as np

fake_input = np.array([[1., 1.],
                       [0., 0.],
                       [1., 1.],
                       [1., 1.]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.InverseMelScale(n_stft=2, n_mels=4)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[0.4181, 0.2522],
#             [0.3385, 0.4193]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.InverseMelScale(n_stft=2, n_mels=4)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[0.81363475 0.38404778]
#       [0.31173563 0.740281  ]]]
```
