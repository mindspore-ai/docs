# Supported Operators

`Ascend` `GPU` `CPU` `Environmental Setup` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_en/supported_operators.md)

<font size=3>**Q: What is the function of the `TransData` operator? Can the performance be optimized?**</font>

A: The `TransData` operator is used in the scenario where the data formats (such as NC1HWC0) used by interconnected operators on the network are inconsistent. In this case, the framework automatically inserts the `TransData` operator to convert the data formats into the same format and then performs computation. You can consider using the `amp` for mixed-precision training. In this way, some `FP32` operations and the invocation of some `TransData` operators can be reduced.

<br/>

<font size=3>**Q: An error occurs when the `Concat` operator concatenates tuples containing multiple tensors. An error occurs when the number of `tensor list` elements entered is greater than or equal to 192. What is a better solution (running in dynamic mode) for `Concat` to concatenate tuples containing multiple Tensors?**</font>

A: The number of tensors to be concatenated at a time cannot exceed 192 according to the bottom-layer specifications of the Ascend operator. You can try to concatenate them twice.

<br/>

<font size=3>**Q: When `Conv2D` is used to define convolution, the `group` parameter is used. Is it necessary to ensure that the value of `group` can be exactly divided by the input and output dimensions? How is the group parameter transferred?**</font>

A: The `Conv2d` operator has the following constraint: When the value of `group` is greater than 1, the value must be the same as the number of input and output channels. Do not use `ops.Conv2D`. Currently, this operator does not support a value of `group` that is greater than 1. Currently, only the `nn.Conv2d` API of MindSpore supports `group` convolution. However, the number of groups must be the same as the number of input and output channels.
The `Conv2D` operator function is as follows:

```python
def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCHW"):
```

If the function contains a `group` parameter, the parameter will be transferred to the C++ layer by default.

<br/>

<font size=3>**Q: Does MindSpore provide 3D convolutional layers?**</font>

A: 3D convolutional layers on Ascend are coming soon. Go to the [Operator List](https://www.mindspore.cn/doc/programming_guide/en/r1.2/operator_list.html) on the official website to view the operators that are supported.

<br/>

<font size=3>**Q: Does MindSpore support matrix transposition?**</font>

A: Yes. For details, see [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose).

<br/>

<font size=3>**Q: Can MindSpore calculate the variance of any tensor?**</font>

A: Currently, MindSpore does not have APIs or operators similar to variance which can directly calculate the variance of a `tensor`. However, MindSpore has sufficient small operators to support such operations. For details, see [class Moments(Cell)](https://www.mindspore.cn/doc/api_python/en/r1.2/_modules/mindspore/nn/layer/math.html#Moments).

<br/>

<font size=3>**Q: Why is data loading abnormal when MindSpore1.0.1 is used in graph data offload mode?**</font>

A: An operator with the `axis` attribute, for example, `ops.Concat(axis=1)((x1, x2))`, is directly used in `construct`. You are advised to initialize the operator in `__init__` as follows:

```python
from mindspore import nn
import mindspore.ops as ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.concat = ops.Concat(axis=1)
    def construct(self, x, y):
        out = self.concat((x, y))
        return out
```

<br/>

<font size=3>**Q: When the `Tile` module in operations executes `__infer__`, the `value` is `None`. Why is the value lost?**</font>

A: The `multiples input` of the `Tile` operator must be a constant. (The value cannot directly or indirectly come from the input of the graph.) Otherwise, the `None` data will be obtained during graph composition because the graph input is transferred only during graph execution and the input data cannot be obtained during graph composition.

<br/>

<font size=3>**Q: Compared with PyTorch, the `nn.Embedding` layer lacks the padding operation. Can other operators implement this operation?**</font>

A: In PyTorch, `padding_idx` is used to set the word vector in the `padding_idx` position in the embedding matrix to 0, and the word vector in the `padding_idx` position is not updated during backward propagation.
In MindSpore, you can manually initialize the weight corresponding to the `padding_idx` position of embedding to 0. In addition, the loss corresponding to `padding_idx` is filtered out through the mask operation during training.

<br/>

<font size=3>**Q: What can I do if the LSTM example on the official website cannot run on Ascend?**</font>

A: Currently, the LSTM runs only on a GPU or CPU and does not support the hardware environment. You can click [MindSpore Operator List](https://www.mindspore.cn/doc/note/en/r1.2/operator_list_ms.html) to view the supported operators.

<br/>

<font size=3>**Q: When conv2d is set to (3,10), Tensor[2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?**</font>

A: This is a TBE operator restriction that the width of x must be greater than that of the kernel. The CPU does not have this operator restriction. Therefore, no error is reported.
