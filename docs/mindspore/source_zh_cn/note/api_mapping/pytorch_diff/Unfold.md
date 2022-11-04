# 比较与torch.nn.unfold的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Unfold.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Unfold

```text
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1) -> Tensor
```

更多内容详见 [torch.nn.Unfold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unfold.html)。

## mindspore.nn.Unfold

```text
class mindspore.nn.Unfold(ksizes, strides, rates, padding='valid')(input_x) -> Tensor
```

更多内容详见 [mindspore.nn.Unfold](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Unfold.html)。

## 差异对比

PyTorch：从批处理输入张量中提取出滑动的局部区域块。输入张量的shape为(N, C, L)，其中N为批处理大小，C为通道数，L表示任意的空间大小，目前仅支持4D输入。输出是shape为(N, C x∏(kernel_size),L)的3维Tensor。

MindSpore：MindSpore此API实现功能与PyTorch功能有差异。PyTorch的kernel_size、stride和dilation支持int和tuple输入，padding支持在输入的两侧添加的隐式零填充。而MindSpore的ksizes、strides和rates三个参数的格式必须是(1, row, col, 1)，padding参数支持两种格式same和valid。MindSpore输入是4维张量，shape为(in_batch, in_depth, in_row, int_col)，输出是shape为(out_batch, out_depth, out_row, out_col)的4维Tensor，其中out_batch和in_batch相同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | input_x | 都是输入4维的Tensor，数据格式为NCHW |
| 参数 | 参数1 | kernel_size | ksizes | 都表示滑动窗口的大小，PyTorch支持int和tuple输入，如果kernel_size是一个int，其值将在所有维度上进行复制；MindSpore支持格式为(1, ksize_row, ksize_col, 1)的tuple或list |
|  | 参数2 | stride      | strides | 都表示空间维度上滑动窗口的步长，PyTorch支持int和tuple输入，默认值是1，如果stride是一个int，其值将在所有维度上进行复制；MindSpore支持格式为(1, stride_row, stride_col, 1)的tuple或list |
| | 参数3 | dilation | rates | dilation表示控制滑动过程中所跨越元素的个数，支持int和tuple输入，默认值是1，如果dilation是一个int，其值将在所有维度上进行复制；rates表示滑窗元素之间的空洞个数，支持格式为(1, rate_row, rate_col, 1)的tuple或list |
| | 参数4 | padding | padding | 都表示填充模式，PyTorch是在输入得两侧添加得隐式零填充，支持int和tuple输入，默认值是0，如果padding是一个int，其值将在所有维度上进行复制；MindSpore支持str输入,可选值有"same"或"valid"，默认值是"valid"，表示所提取的区域块被原始输入所覆盖，取值为"same"时表示所提取的区域块的部分区域可以在原始输入之外进行零填充 |
| 输出 | 单输出 | output | output | PyTorch输出3维张量，MindSpore输出4维张量 |

### 代码示例1

> PyTorch的stride默认值是1，dilation默认值是1，padding默认值是0，由于是输入是4维Tensor且这三个参数默认值都是int，将在所有维度上进行复制。为了得到与PyTorch相同的结果，MindSpore首先分别将Unfold算子的strides、rates和padding分别设置为(1, 1, 1, 1)、(1, 1, 1, 1)和"valid"，若kernel_size为一个int时即kernel_size=a时，将ksizes设置为(1, a, a, 1);若kernel_size为一个tuple时即kernel_size=(a,b)时，将ksizes设置为(1, a, b, 1)，其次为了输出结果完全一致，首先将MindSpore输出结果进行Reshape操作，然后通过下面的操作进行Concat得到最终结果。

```python
# PyTorch
import torch
from torch import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
torch_unfold = torch.nn.Unfold(kernel_size=(2, 2), dilation=1, padding=0, stride=1)
torch_output = torch_unfold(input)
# torch_output.shape=(1, 8, 9)
print(torch_output.numpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
ms_unfold = mindspore.nn.Unfold(ksizes=(1, 2, 2, 1), rates=(1, 1, 1, 1), padding="valid", strides=(1, 1, 1, 1))
ms_output = ms_unfold(input)
# ms_output.shape = (1, 8, 3, 3)
print(ms_output.asnumpy())
# [[[[ 0.  1.  2.]
#    [ 4.  5.  6.]
#    [ 8.  9. 10.]]
#
#   ...
#
#   [[21. 22. 23.]
#    [25. 26. 27.]
#    [29. 30. 31.]]]]
out_batch, out_depth, out_row, out_col = ms_output.shape
ms_reshape = mindspore.ops.Reshape()
ms_output = ms_reshape(ms_output, (out_batch, out_depth, out_row * out_col))
# ms_output.shape = (1, 8, 9)
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]
ms_concat = mindspore.ops.Concat()
output = None
for i in range(out_batch):
    odd = None
    even = None
    for j in range(out_depth):
        data = ms_output[i,j,:]
        data = ms_reshape(data, (1, data.shape[0]))
        if j % 2 == 0:
            if even is None:
                even = data
            else:
                even = ms_concat((even, data))
        else:
            if odd is None:
                odd = data
            else:
                odd = ms_concat((odd, data))
    temp = ms_concat((even, odd))
    temp = ms_reshape(temp, (1, temp.shape[0], temp.shape[1]))
    if i == 0:
        output = temp
    else:
        output = ms_concat((output, temp))
ms_output = output
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]
```

### 代码示例2

> 在PyTorch的padding参数取默认值0，MindSpore的padding取默认值"valid"前提下，当其余三个参数相对应设置时，若kernel_size为一个int时即kernel_size=a时，将ksizes设置为(1, a, a, 1);若kernel_size为一个tuple时即kernel_size=(a,b)时，将ksizes设置为(1, a, b, 1)；若stride为一个int时即stride=a时，将strides设置为(1, a, a, 1);若stride为一个tuple时即stride=(a,b)时，将strides设置为(1, a, b, 1)；若dilation为一个int时即dilation=a时，将rates设置为(1, a, a, 1)；若dilation为一个tuple时即dilation=(a,b)时，将rates设置为(1, a, b, 1)。其次为了输出结果完全一致，首先将MindSpore输出结果进行Reshape操作，然后通过下面的操作进行Concat得到最终结果。

```python
# PyTorch
import torch
from torch import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
torch_unfold = torch.nn.Unfold(kernel_size=(2,2), dilation=(1, 1), padding=0, stride=(1, 1))
torch_output = torch_unfold(input)
# torch_output.shape=(1, 8, 9)
print(torch_output.numpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
ms_unfold = mindspore.nn.Unfold(ksizes=(1, 2, 2, 1), rates=(1, 1, 1, 1), padding="valid", strides=(1, 1, 1, 1))
ms_output = ms_unfold(input)
# ms_output.shape = (1, 8, 3, 3)
print(ms_output.asnumpy())
# [[[[ 0.  1.  2.]
#    [ 4.  5.  6.]
#    [ 8.  9. 10.]]
#
#   ...
#
#   [[21. 22. 23.]
#    [25. 26. 27.]
#    [29. 30. 31.]]]]
out_batch, out_depth, out_row, out_col = ms_output.shape
ms_reshape = mindspore.ops.Reshape()
ms_output = ms_reshape(ms_output, (out_batch, out_depth, out_row * out_col))
# ms_output.shape = (1, 8, 9)
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]
ms_concat = mindspore.ops.Concat()
output = None
for i in range(out_batch):
    odd = None
    even = None
    for j in range(out_depth):
        data = ms_output[i,j,:]
        data = ms_reshape(data, (1, data.shape[0]))
        if j % 2 == 0:
            if even is None:
                even = data
            else:
                even = ms_concat((even, data))
        else:
            if odd is None:
                odd = data
            else:
                odd = ms_concat((odd, data))
    temp = ms_concat((even, odd))
    temp = ms_reshape(temp, (1, temp.shape[0], temp.shape[1]))
    if i == 0:
        output = temp
    else:
        output = ms_concat((output, temp))
ms_output = output
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]
```

### 代码示例3

> PyTorch的padding表示输入得两侧添加得隐式零填充，支持int和tuple输入，默认值是0，与MindSpore的padding取默认值"valid"相对应。当PyTorch的padding取其他值时，MindSpore的padding的可取值只有"valid"和"same"，所以没有与之对应的取值，故输出结果不一致。

```python
# PyTorch
import torch
from torch import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
torch_unfold = torch.nn.Unfold(kernel_size=(2,2), dilation=1, padding=1, stride=1)
torch_output = torch_unfold(input)
# ms_output.shape = (1, 8, 25)
print(torch_output.numpy())
# [[[ 0.  0.  0.  0.  0.  0.  0.  1.  2.  3.  0.  4.  5.  6.  7.  0.  8.
#     9. 10. 11.  0. 12. 13. 14. 15.]
#   [ 0.  0.  0.  0.  0.  0.  1.  2.  3.  0.  4.  5.  6.  7.  0.  8.  9.
#    10. 11.  0. 12. 13. 14. 15.  0.]
#   [ 0.  0.  1.  2.  3.  0.  4.  5.  6.  7.  0.  8.  9. 10. 11.  0. 12.
#    13. 14. 15.  0.  0.  0.  0.  0.]
#   [ 0.  1.  2.  3.  0.  4.  5.  6.  7.  0.  8.  9. 10. 11.  0. 12. 13.
#    14. 15.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0. 16. 17. 18. 19.  0. 20. 21. 22. 23.  0. 24.
#    25. 26. 27.  0. 28. 29. 30. 31.]
#   [ 0.  0.  0.  0.  0. 16. 17. 18. 19.  0. 20. 21. 22. 23.  0. 24. 25.
#    26. 27.  0. 28. 29. 30. 31.  0.]
#   [ 0. 16. 17. 18. 19.  0. 20. 21. 22. 23.  0. 24. 25. 26. 27.  0. 28.
#    29. 30. 31.  0.  0.  0.  0.  0.]
#   [16. 17. 18. 19.  0. 20. 21. 22. 23.  0. 24. 25. 26. 27.  0. 28. 29.
#    30. 31.  0.  0.  0.  0.  0.  0.]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor
# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

input = Tensor(np.arange(32).reshape((1, 2, 4, 4)).astype(np.float32))
ms_unfold = mindspore.nn.Unfold(ksizes=(1, 2, 2, 1), rates=(1, 1, 1, 1), padding="same", strides=(1, 1, 1, 1))
ms_output = ms_unfold(input)
# ms_output.shape = (1, 8, 4, 4)
print(ms_output.asnumpy())
[[[[ 0.  1.  2.  3.]
#    [ 4.  5.  6.  7.]
#    [ 8.  9. 10. 11.]
#    [12. 13. 14. 15.]]
#
#   ...
#
#   [[21. 22. 23.  0.]
#    [25. 26. 27.  0.]
#    [29. 30. 31.  0.]
#    [ 0.  0.  0.  0.]]]]
out_batch, out_depth, out_row, out_col = ms_output.shape
ms_reshape = mindspore.ops.Reshape()
ms_output = ms_reshape(ms_output, (out_batch, out_depth, out_row * out_col))
# ms_output.shape = (1, 8, 16)
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  4.  5.  6.  8.  9. 10.]
#   [16. 17. 18. 20. 21. 22. 24. 25. 26.]
#   [ 1.  2.  3.  5.  6.  7.  9. 10. 11.]
#   [17. 18. 19. 21. 22. 23. 25. 26. 27.]
#   [ 4.  5.  6.  8.  9. 10. 12. 13. 14.]
#   [20. 21. 22. 24. 25. 26. 28. 29. 30.]
#   [ 5.  6.  7.  9. 10. 11. 13. 14. 15.]
#   [21. 22. 23. 25. 26. 27. 29. 30. 31.]]]
ms_concat = mindspore.ops.Concat()
output = None
for i in range(out_batch):
    odd = None
    even = None
    for j in range(out_depth):
        data = ms_output[i,j,:]
        data = ms_reshape(data, (1, data.shape[0]))
        if j % 2 == 0:
            if even is None:
                even = data
            else:
                even = ms_concat((even, data))
        else:
            if odd is None:
                odd = data
            else:
                odd = ms_concat((odd, data))
    temp = ms_concat((even, odd))
    temp = ms_reshape(temp, (1, temp.shape[0], temp.shape[1]))
    if i == 0:
        output = temp
    else:
        output = ms_concat((output, temp))
ms_output = output
print(ms_output.asnumpy())
# [[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
#   [ 1.  2.  3.  0.  5.  6.  7.  0.  9. 10. 11.  0. 13. 14. 15.  0.]
#   [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.  0.  0.  0.  0.]
#   [ 5.  6.  7.  0.  9. 10. 11.  0. 13. 14. 15.  0.  0.  0.  0.  0.]
#   [16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31.]
#   [17. 18. 19.  0. 21. 22. 23.  0. 25. 26. 27.  0. 29. 30. 31.  0.]
#   [20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31.  0.  0.  0.  0.]
#   [21. 22. 23.  0. 25. 26. 27.  0. 29. 30. 31.  0.  0.  0.  0.  0.]]]
```
