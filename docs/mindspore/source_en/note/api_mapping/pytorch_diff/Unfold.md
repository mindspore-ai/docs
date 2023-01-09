# Function Differences with torch.nn.Unfold

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Unfold.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Unfold

```text
torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)(input) -> Tensor
```

For more information, see [torch.nn.Unfold](https://pytorch.org/docs/1.8.1/generated/torch.nn.Unfold.html).

## mindspore.nn.Unfold

```text
class mindspore.nn.Unfold(ksizes, strides, rates, padding='valid')(x) -> Tensor
```

For more information, see [mindspore.nn.Unfold](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Unfold.html).

## Differences

PyTorch: Extracts the sliding local area blocks from the batch input tensor. The shape of the input tensor is (N, C, H, W), where N is the batch size, C is the number of channels, H is the height, and W is the width. The output is a three-dimensional Tensor.

MindSpore: Implementation function of API in MinSpore differs from that of PyTorch. kernel_size, stride, and dilation of PyTorch support int and tuple inputs, and padding supports implicit zero padding added on either side of the input, while MindSpore ksizes, strides and rates three parameters must be the format of (1, row, col, 1), and padding parameters support two formats same and valid. MindSpore input is a four-dimensional tensor with shape (in_batch, in_depth, in_row, int_col), and output is a four-dimensional Tensor with shape (out_batch, out_depth, out_row, out_col), where out_batch and in_batch are the same.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Input | Single input | input | x | Both are input four-dimensional Tensor with data format NCHW|
| Parameters | Parameter 1 | kernel_size | ksizes | The function is the same, but the input format is not. Both indicate the size of the sliding window. PyTorch supports int and tuple inputs. If kernel_size is an int, its value will be replicated in all dimensions. MindSpore supports tuple or list of the format (1, ksize_row, ksize_col, 1) |
| | Parameter 2 | dilation | rates | The function is the same, but the input format is not. dilation indicates the number of elements spanned during the control slide, supports int and tuple input, and default value is 1. If dilation is an int, its value will be replicated in all dimensions. rates indicates the number of holes between slide elements, supports tuple or list of the format (1, rate_row, rate_col, 1) |
| | Parameter 3 | padding | padding | The function is the same. Indicating padding mode, PyTorch is zero padding on both sides of the input and supports int and tuple inputs, and the default value is 0. If padding is an int, its value will be copied in all dimensions. MindSpore supports str input, with optional values "same" or "valid", and the default value is "valid", which means the extracted region block is covered by the original input. The value "same" means part of the extracted region block can perform zero padding outside the original input. |
|  | Parameter 4 | stride      | strides | The function is the same, but the input format is not. Both indicate the step size of the sliding window in the spatial dimension. PyTorch supports int and tuple inputs, with a default value of 1. If stride is an int, its value will be replicated in all dimensions. MindSpore supports tuples or lists of the format (1, stride_row, stride_col, 1). |

### Code Example 1

> PyTorch stride defaults to 1, dilation defaults to 1, and padding defaults to 0. Since the input is a four-dimensional Tensor and the default values of all three parameters are int, they will be replicated in all dimensions. To obtain the same results as PyTorch, MindSpore first sets the strides, rates and padding of the Unfold operator to (1, 1, 1, 1), (1, 1, 1, 1) and "valid", respectively. Set ksizes to (1, a, a, 1) if kernel_size is an int, i.e., if kernel_size=a. If kernel_size is a tuple, i.e. kernel_size=(a,b), set ksizes to (1, a, b, 1), and secondly, in order to have the same output result, the output result of MindSpore will be Reshape, and then the final result will be obtained by the following operation of Concat.

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

#   [[16. 17. 18.]
#    [20. 21. 22.]
#    [24. 25. 26.]]

#   [[ 1.  2.  3.]
#    [ 5.  6.  7.]
#    [ 9. 10. 11.]]

#   [[17. 18. 19.]
#    [21. 22. 23.]
#    [25. 26. 27.]]

#   [[ 4.  5.  6.]
#    [ 8.  9. 10.]
#    [12. 13. 14.]]

#   [[20. 21. 22.]
#    [24. 25. 26.]
#    [28. 29. 30.]]

#   [[ 5.  6.  7.]
#    [ 9. 10. 11.]
#    [13. 14. 15.]]

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

### Code Example 2

> With the default value of 0 for PyTorch padding parameter and "valid" for MindSpore padding, set ksizes to (1, a, a, 1) if kernel_size is an int, i.e. kernel_size=a, when the remaining three parameters are set correspondingly. If kernel_size is a tuple, i.e. kernel_size=(a,b), set ksizes to (1, a, b, 1). If stride is an int, i.e. stride=a, set strides to (1, a, a, 1). If stride is a tuple, i.e. stride=(a,b), set strides to (1, a, b, 1). If dilation is an int, i.e. dilation=a, set rates to (1, a, a, 1); and if dilation is a tuple, i.e. dilation=(a,b), set rates to (1, a, b, 1). Secondly, in order to get the same output result, the output of MindSpore is Reshape, and then the final result is Concat by the following operation.

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

#   [[16. 17. 18.]
#    [20. 21. 22.]
#    [24. 25. 26.]]

#   [[ 1.  2.  3.]
#    [ 5.  6.  7.]
#    [ 9. 10. 11.]]

#   [[17. 18. 19.]
#    [21. 22. 23.]
#    [25. 26. 27.]]

#   [[ 4.  5.  6.]
#    [ 8.  9. 10.]
#    [12. 13. 14.]]

#   [[20. 21. 22.]
#    [24. 25. 26.]
#    [28. 29. 30.]]

#   [[ 5.  6.  7.]
#    [ 9. 10. 11.]
#    [13. 14. 15.]]

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

### Code Example 3

> PyTorch padding means zero padding on both sides of the input, and supports both int and tuple inputs. The default value is 0, which corresponds to MindSpore padding taking the default value of "valid". When PyTorch padding takes other values, and MindSpore padding only has the values "valid" and "same", there is no corresponding value, so the output is inconsistent.

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
# [[[[ 0.  1.  2.  3.]
#    [ 4.  5.  6.  7.]
#    [ 8.  9. 10. 11.]
#    [12. 13. 14. 15.]]

#   [[16. 17. 18. 19.]
#    [20. 21. 22. 23.]
#    [24. 25. 26. 27.]
#    [28. 29. 30. 31.]]

#   [[ 1.  2.  3.  0.]
#    [ 5.  6.  7.  0.]
#    [ 9. 10. 11.  0.]
#    [13. 14. 15.  0.]]

#   [[17. 18. 19.  0.]
#    [21. 22. 23.  0.]
#    [25. 26. 27.  0.]
#    [29. 30. 31.  0.]]

#   [[ 4.  5.  6.  7.]
#    [ 8.  9. 10. 11.]
#    [12. 13. 14. 15.]
#    [ 0.  0.  0.  0.]]

#   [[20. 21. 22. 23.]
#    [24. 25. 26. 27.]
#    [28. 29. 30. 31.]
#    [ 0.  0.  0.  0.]]

#   [[ 5.  6.  7.  0.]
#    [ 9. 10. 11.  0.]
#    [13. 14. 15.  0.]
#    [ 0.  0.  0.  0.]]

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
