# Differences with torchaudio.transforms.AmplitudeToDB

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AmplitudeToDB.md)

## torchaudio.transforms.AmplitudeToDB

```python
class torchaudio.transforms.AmplitudeToDB(stype: str = 'power', top_db: Optional[float] = None)
```

For more information, see [torchaudio.transforms.AmplitudeToDB](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.AmplitudeToDB.html).

## mindspore.dataset.audio.AmplitudeToDB

```python
class mindspore.dataset.audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=80.0)
```

For more information, see [mindspore.dataset.audio.AmplitudeToDB](https://mindspore.cn/docs/en/br_base/api_python/dataset_audio/mindspore.dataset.audio.AmplitudeToDB.html#mindspore.dataset.audio.AmplitudeToDB).

## Differences

PyTorch: Turn the input audio waveform from the amplitude/power scale to decibel scale.

MindSpore: Turn the input audio waveform from the amplitude/power scale to decibel scale. Specified lower bound of the input waveform and multiplier reference value for db are supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | stype    | stype    | - |
|     | Parameter2 | top_db   | top_db   | Default value is different |
|     | Parameter3 | -   | ref_value   | Multiplier reference value for generating db_multiplier, PyTorch uses 1e-10 internally |
|     | Parameter4 | -   | amin   | Lower bound to clamp the input waveform, PyTorch uses 1.0 internally |

## Code Example

```python
import numpy as np

fake_input = np.array([[[[-0.2197528, 0.3821656]]],
                       [[[0.57418776, 0.46741104]]],
                       [[[0.76986176, -0.5793846]]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.AmplitudeToDB(stype="power", top_db=80.)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[-84.1775,  -4.1775]]],
#        [[[ -2.4095,  -3.3030]]],
#        [[[ -1.1359, -81.1359]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.AmplitudeToDB(stype=audio.ScaleType.POWER, top_db=80., ref_value=1.0, amin=1e-10)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[-84.17748    -4.177484 ]]]
#      [[[ -2.4094608  -3.3030105]]]
#      [[[ -1.1358725 -81.13587  ]]]]
```
