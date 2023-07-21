# Supported Operators

`Ascend` `GPU` `CPU` `Environmental Setup` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_en/supported_operators.md)

<font size=3>**Q: Why is data loading abnormal when MindSpore1.0.1 is used in graph data offload mode?**</font>

A: An operator with the `axis` attribute, for example, `P.Concat(axis=1)((x1, x2))`, is directly used in `construct`. You are advised to initialize the operator in `__init__` as follows:

```python
from mindspore import nn
from mindspore.ops import operations as P

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.concat = P.Concat(axis=1)
    def construct(self, x, y):
        out = self.concat((x, y))
        return out
```

<br/>

<font size=3>**Q: When the `Tile` module in operations executes `__infer__`, the `value` is `None`. Why is the value lost?**</font>

A: The `multiples input` of the `Tile` operator must be a constant. (The value cannot directly or indirectly come from the input of the graph.) Otherwise, the `None` data will be obtained during graph composition because the graph input is transferred only during graph execution and the input data cannot be obtained during graph composition.
For details, see "Other Constraints" in the [Constraints on Network Construction](https://www.mindspore.cn/doc/note/en/r1.0/constraints_on_network_construction.html).

<br/>

<font size=3>**Q: Compared with PyTorch, the `nn.Embedding` layer lacks the padding operation. Can other operators implement this operation?**</font>

A: In PyTorch, `padding_idx` is used to set the word vector in the `padding_idx` position in the embedding matrix to 0, and the word vector in the `padding_idx` position is not updated during backward propagation.
In MindSpore, you can manually initialize the weight corresponding to the `padding_idx` position of embedding to 0. In addition, the loss corresponding to `padding_idx` is filtered out through the mask operation during training.

<br/>

<font size=3>**Q: What can I do if the LSTM example on the official website cannot run on Ascend?**</font>

A: Currently, the LSTM runs only on a GPU or CPU and does not support the hardware environment. You can click [here](https://www.mindspore.cn/doc/note/en/r1.0/operator_list_ms.html) to view the supported operators.

<br/>

<font size=3>**Q: When conv2d is set to (3,10), Tensor[2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?**</font>

A: This is a TBE operator restriction that the width of x must be greater than that of the kernel. The CPU does not have this operator restriction. Therefore, no error is reported.