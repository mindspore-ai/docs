# 算子支持类

`Ascend` `CPU` `GPU` `环境准备` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_zh_cn/supported_operators.md)

<font size=3>**Q：`TransData`算子的功能是什么，能否优化性能？**</font>

A：`TransData`算子出现的场景是：如果网络中相互连接的算子使用的数据格式不一致（如NC1HWC0），框架就会自动插入`transdata`算子使其转换成一致的数据格式，然后再进行计算。 可以考虑训练的时候用我们的`amp`做混合精度，这样能减少一些`fp32`的运算，应该能减少一些`transdata`算子的调用。

<br/>

<font size=3>**Q：算子`Concat`拼接包含多个Tensor的元组出错，似乎传入的`tensor list`元素个数>=192就会报错。如果要`Concat`包含多个Tensor的元组，有什么较好的解决方案？**</font>

A：这个昇腾算子底层规格限制一次拼接的Tensor个数不能超过192个，可以尝试分开两次进行拼接。

<br/>

<font size=3>**Q：在使用`Conv2D`进行卷积定义的时候使用到了`group`的参数，`group`的值不是只需要保证可以被输入输出的维度整除即可了吗？`group`参数的传递方式是怎样的呢？**</font>

A：`Conv2D`算子是有这个约束条件的：当`group`大于1 时，其值必须要与输入输出的通道数相等。不要使用`ops.Conv2D`，这个算子目前不支持`group`>1。目前MindSpore只有`nn.Conv2D`接口支持组卷积，但是有`group`要与输入输出的通道数相等的约束。
`Conv2D`算子的

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

函数中带有`group`参数，这个参数默认就会被传到C++层。

<br/>

<font size=3>**Q：Convolution Layers有没有提供3D卷积？**</font>

A：目前MindSpore在Ascend上有支持3D卷积的计划。您可以关注官网的[支持列表](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.2/operator_list.html)，等到算子支持后会在表中展示。

<br/>

<font size=3>**Q：MindSpore支持矩阵转置吗？**</font>

A：支持，请参考`mindspore.ops.Transpose`的[算子教程](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose)。

<br/>

<font size=3>**Q：请问MindSpore能算给定任意一个`tensor`的方差吗？**</font>

A：MindSpore目前暂无可以直接求出`tensor`方差的算子或接口。不过MindSpore有足够多的小算子可以支持用户实现这样的操作，你可以参考[class Moments(Cell)](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/_modules/mindspore/nn/layer/math.html#Moments)来实现。

<br/>

<font size=3>**Q：使用MindSpore-1.0.1版本在图数据下沉模式加载数据异常是什么原因？**</font>

A：应该是`construct`中直接使用了带有`axis`属性的算子，比如`ops.Concat(axis=1)((x1, x2))`这种，建议把算子在`__init__`中初始化 像这样

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

<font size=3>**Q：`nn.Embedding`层与PyTorch相比缺少了`Padding`操作，有其余的算子可以实现吗？**</font>

A：在PyTorch中`padding_idx`的作用是将embedding矩阵中`padding_idx`位置的词向量置为0，并且反向传播时不会更新`padding_idx`位置的词向量。在MindSpore中，可以手动将embedding的`padding_idx`位置对应的权重初始化为0，并且在训练时通过`mask`的操作，过滤掉`padding_idx`位置对应的`Loss`。

<br/>

<font size=3>**Q：Operations中`Tile`算子执行到`__infer__`时`value`值为`None`，丢失了数值是怎么回事？**</font>

A：`Tile`算子的`multiples input`必须是一个常量（该值不能直接或间接来自于图的输入）。否则构图的时候会拿到一个`None`的数据，因为图的输入是在图执行的时候才传下去的，构图的时候拿不到图的输入数据。
相关的资料可以看[静态图语法支持](https://www.mindspore.cn/doc/note/zh-CN/r1.2/static_graph_syntax_support.html)。

<br/>

<font size=3>**Q：官网的LSTM示例在Ascend上跑不通。**</font>

A：目前LSTM只支持在GPU和CPU上运行，暂不支持硬件环境，您可以通过[MindSpore算子支持列表](https://www.mindspore.cn/doc/note/zh-CN/r1.2/operator_list_ms.html)查看算子支持情况。

<br/>

<font size=3>**Q：conv2d设置为(3,10),Tensor[2,2,10,10]，在ModelArts上利用Ascend跑，报错：`FM_W+pad_left+pad_right-KW>=strideW`，CPU下不报错。**</font>

A：这是TBE这个算子的限制，x的width必须大于kernel的width。CPU的这个算子没有这个限制，所以不报错。
