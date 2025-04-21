# 比较与torchaudio.transforms.InverseMelScale的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/InverseMelScale.md)

## torchaudio.transforms.InverseMelScale

```python
class torchaudio.transforms.InverseMelScale(n_stft: int, n_mels: int = 128, sample_rate: int = 16000, f_min: float = 0.0, f_max: Optional[float] = None,
                                            max_iter: int = 100000, tolerance_loss: float = 1e-05, tolerance_change: float = 1e-08, sgdargs: Optional[dict] = None,
                                            norm: Optional[str] = None)
```

更多内容详见[torchaudio.transforms.InverseMelScale](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.InverseMelScale.html)。

## mindspore.dataset.audio.InverseMelScale

```python
class mindspore.dataset.audio.InverseMelScale(n_stft, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None,
                                              max_iter=100000, tolerance_loss=1e-5, tolerance_change=1e-8, sgdargs=None,
                                              norm=NormType.NONE, mel_type=MelType.HTK)
```

更多内容详见[mindspore.dataset.audio.InverseMelScale](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.InverseMelScale.html#mindspore.dataset.audio.InverseMelScale)。

## 差异对比

PyTorch：使用转换矩阵从梅尔频率STFT求解普通频率的STFT。

MindSpore：使用转换矩阵从梅尔频率STFT求解普通频率的STFT，支持指定梅尔频谱的尺度。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | n_stft     | n_stft     | - |
|     | 参数2 | n_mels    | n_mels    | - |
|     | 参数3 | sample_rate  | sample_rate  | - |
|     | 参数4 | f_min  | f_min    | - |
|     | 参数5 | f_max   | f_max     | - |
|     | 参数6 | max_iter   | max_iter     | - |
|     | 参数7 | tolerance_loss   | tolerance_loss     | - |
|     | 参数8 | tolerance_change   | tolerance_change     | - |
|     | 参数9 | sgdargs   | sgdargs     | - |
|     | 参数10 | norm   | norm     | - |
|     | 参数11 | -   | mel_type      | 要使用的Mel尺度 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[1., 1.],
                       [0., 0.],
                       [1., 1.],
                       [1., 1.]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T
torch.manual_seed(1)

transformer = T.InverseMelScale(n_stft=2, n_mels=4)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[0.7576, 0.4031],
#              [0.2793, 0.7347]])

# MindSpore
import mindspore as ms
import mindspore.dataset.audio as audio
ms.dataset.config.set_seed(3)

transformer = audio.InverseMelScale(n_stft=2, n_mels=4)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[0.5507979  0.07072488]
#        [0.7081478  0.8399491 ]]]
```
