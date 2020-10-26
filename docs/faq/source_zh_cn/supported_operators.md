# 算子支持类

`Ascend` `CPU` `GPU` `环境准备` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_zh_cn/supported_operators.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q：`nn.Embedding`层与PyTorch相比缺少了`Padding`操作，有其余的算子可以实现吗？

A：在PyTorch中`padding_idx`的作用是将embedding矩阵中`padding_idx`位置的词向量置为0，并且反向传播时不会更新`padding_idx`位置的词向量。在MindSpore中，可以手动将embedding的`padding_idx`位置对应的权重初始化为0，并且在训练时通过`mask`的操作，过滤掉`padding_idx`位置对应的`Loss`。

<br/>

Q：Operations中`Tile`算子执行到`__infer__`时`value`值为`None`，丢失了数值是怎么回事？

A：`Tile`算子的`multiples input`必须是一个常量（该值不能直接或间接来自于图的输入）。否则构图的时候会拿到一个`None`的数据，因为图的输入是在图执行的时候才传下去的，构图的时候拿不到图的输入数据。
相关的资料可以看[相关文档](https://www.mindspore.cn/doc/note/zh-CN/r1.0/constraints_on_network_construction.html)的“其他约束”。

<br/>

Q：官网的LSTM示例在Ascend上跑不通

A：目前LSTM只支持在GPU和CPU上运行，暂不支持硬件环境，您可以[点击这里](https://www.mindspore.cn/doc/note/zh-CN/r1.0/operator_list_ms.html)查看算子支持情况。

<br/>

Q：conv2d设置为(3,10),Tensor[2,2,10,10]，在ModelArts上利用Ascend跑，报错：`FM_W+pad_left+pad_right-KW>=strideW`，CPU下不报错。

A：这是TBE这个算子的限制，x的width必须大于kernel的width。CPU的这个算子没有这个限制，所以不报错。