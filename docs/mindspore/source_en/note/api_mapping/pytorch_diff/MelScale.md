# Differences with torchaudio.transforms.MelScale

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MelScale.md)

## torchaudio.transforms.MelScale

```python
class torchaudio.transforms.MelScale(n_mels: int = 128, sample_rate: int = 16000, f_min: float = 0.0, f_max: Optional[float] = None,
                                     n_stft: Optional[int] = None, norm: Optional[str] = None)
```

For more information, see [torchaudio.transforms.MelScale](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.MelScale.html).

## mindspore.dataset.audio.MelScale

```python
class mindspore.dataset.audio.MelScale(n_mels=128, sample_rate=16000, f_min=0.0, f_max=None,
                                       n_stft=201, norm=NormType.NONE, mel_type=MelType.HTK)
```

For more information, see [mindspore.dataset.audio.MelScale](https://mindspore.cn/docs/en/br_base/api_python/dataset_audio/mindspore.dataset.audio.MelScale.html#mindspore.dataset.audio.MelScale).

## Differences

PyTorch: Convert normal STFT to STFT at the Mel scale.

MindSpore: Convert normal STFT to STFT at the Mel scale.. Mel scale can be specified.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | n_mels     | n_mels     | - |
|     | Parameter2 | sample_rate     | sample_rate     | - |
|     | Parameter4 | f_min  | f_min    | - |
|     | Parameter5 | f_max   | f_max     | - |
|     | Parameter6 | n_stft    | n_stft     | - |
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

transformer = T.MelScale(n_stft=4, n_mels=2)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[0.0000, 0.0000],
#              [0.5394, 0.5394]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.MelScale(n_stft=4, n_mels=2)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[0.         0.        ]
#       [0.53936154 0.53936154]]
```
