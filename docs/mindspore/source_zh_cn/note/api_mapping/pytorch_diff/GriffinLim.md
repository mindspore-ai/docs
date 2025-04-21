# 比较与torchaudio.transforms.GriffinLim的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GriffinLim.md)

## torchaudio.transforms.GriffinLim

```python
class torchaudio.transforms.GriffinLim(n_fft: int = 400, n_iter: int = 32, win_length: Optional[int] = None, hop_length: Optional[int] = None,
                                       window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>, power: float = 2.0,
                                       normalized: bool = False, wkwargs: Optional[dict] = None, momentum: float = 0.99,
                                       length: Optional[int] = None, rand_init: bool = True)
```

更多内容详见[torchaudio.transforms.GriffinLim](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.GriffinLim.html)。

## mindspore.dataset.audio.GriffinLim

```python
class mindspore.dataset.audio.GriffinLim(n_fft=400, n_iter=32, win_length=None, hop_length=None,
                                         window_type=WindowType.HANN, power=2.0,
                                         momentum=0.99, length=None, rand_init=True)
```

更多内容详见[mindspore.dataset.audio.GriffinLim](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.GriffinLim.html#mindspore.dataset.audio.GriffinLim)。

## 差异对比

PyTorch：使用Griffin-Lim算法从线性幅度频谱图中计算信号波形。支持自定义窗函数或对窗函数传入不同的配置参数。支持对STFT结果进行幅值规范化。

MindSpore：使用Griffin-Lim算法从线性幅度频谱图中计算信号波形。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | n_fft     | n_fft     | - |
|     | 参数2 | n_iter    | n_iter    | - |
|     | 参数3 | win_length  | win_length    | - |
|     | 参数4 | hop_length  | hop_length    | - |
|     | 参数5 | window_fn   | window_type   | MindSpore仅支持5种窗函数 |
|     | 参数6 | power  | power    | - |
|     | 参数7 | normalized  | -    | STFT后幅值规范化，MindSpore不支持 |
|     | 参数8 | wkwargs   | -     | 自定义窗函数的入参，MindSpore不支持 |
|     | 参数9 | momentum   | momentum     | - |
|     | 参数10 | length   | length     | - |
|     | 参数11 | rand_init  | rand_init     | - |

## 代码示例

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
