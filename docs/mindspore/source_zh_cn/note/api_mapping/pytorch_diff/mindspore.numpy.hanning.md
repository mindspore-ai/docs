# 比较与torch.hann_window的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.numpy.hanning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.hann_window

```text
torch.hann_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

更多内容详见[torch.hann_window](https://pytorch.org/docs/1.8.1/generated/torch.hann_window.html)。

## mindspore.numpy.hanning

```text
mindspore.numpy.hanning(M) -> Tensor
```

更多内容详见[mindspore.numpy.hanning](https://mindspore.cn/docs/zh-CN/master/api_python/numpy/mindspore.numpy.hanning.html)。

## 差异对比

PyTorch：返回size与window_length相同的汉宁窗，periodic参数确定返回窗口是否会删除对称窗口的最后一个重复值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但缺少参数periodic，功能实现相当于设置periodic为False。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 |window_length | M | 功能一致，参数名不同 |
|参数 | 参数1 | periodic | -    | MindSpore中相当于设置为False |
|  | 参数2 | dtype        | -    | MindSpore无此参数，输出dtype为Float32，与标杆默认一致 |
| | 参数3 | layout | - | 不涉及 |
| | 参数4 | device | - | 不涉及 |
| | 参数5 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |

### 代码示例1

> PyTorch算子中periodic参数确定返回窗口是否会删除对称窗口的最后一个重复值，而MindSpore算子缺少改参数，当于设置periodic为False。

```python
# PyTorch
import torch

torch_output = torch.hann_window(12, periodic=False)
print(torch_output.numpy())
# [0.         0.07937324 0.29229248 0.57115734 0.82743037 0.97974646
#  0.9797465  0.8274305  0.5711575  0.29229265 0.07937327 0.        ]

# MindSpore
import mindspore

ms_output = mindspore.numpy.hanning(12)
print(ms_output)
# [0.         0.07937324 0.29229248 0.57115734 0.8274303  0.97974694
#  0.97974706 0.8274305  0.5711576  0.29229274 0.07937327 0.        ]
```
