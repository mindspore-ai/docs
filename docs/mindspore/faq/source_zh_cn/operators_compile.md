﻿# 算子编译

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/faq/source_zh_cn/operators_compile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

<font size=3>**Q: 在使用`ops.concat`算子时，因为数据规模有点大，导致报错`Error:Input and (output + workspace) num should <=192!`，可以怎么处理？**</font>

A: 这种报错，主要为`ops.concat`算子提示`shape`过大。建议对`dataset`对象创建迭代器时可设置输出为`numpy`, 如下设置：

```python
gallaryloader.create_dict_iterator(output_numpy=True)
```

另外在上述后处理环节（非网络计算过程中，即非`construct`函数里面），可以采用`numpy`直接计算，如采用`numpy.concatenate`代替上述`ops.concat`进行计算。

<br/>

<font size=3>**Q: 请问在静态图模式的`construct`函数里，如何把一个`tensor`中所含有的负数值全部去除掉？**</font>

A: 建议使用`ops.clip_by_value`接口，把负数全变成0来进行计算。

<br/>

<font size=3>**Q: `TransData`算子的功能是什么，能否优化性能？**</font>

A: `TransData`算子出现的场景是: 如果网络中相互连接的算子使用的数据格式不一致（如NC1HWC0），框架就会自动插入`transdata`算子使其转换成一致的数据格式，然后再进行计算。 华为Ascend支持5D格式运算，通过`transdata`算子将数据由4D转为5D以提升性能。

<br/>

<font size=3>**Q: 算子`Concat`拼接包含多个Tensor的元组出错，似乎传入的`tensor list`元素个数>=192就会报错。如果要`Concat`包含多个Tensor的元组，有什么较好的解决方案？**</font>

A: 这个昇腾算子底层规格限制一次拼接的Tensor个数不能超过192个，可以尝试分开两次进行拼接。

<br/>

<font size=3>**Q: 在使用`Conv2D`进行卷积定义的时候使用到了`group`的参数，`group`的值不是只需要保证可以被输入输出的维度整除即可了吗？`group`参数的传递方式是怎样的呢？**</font>

A: `Conv2D`算子是有这个约束条件的: 当`group`大于1 时，其值必须要与输入输出的通道数相等。不要使用`ops.Conv2D`，这个算子目前不支持`group`>1。目前MindSpore只有`nn.Conv2D`接口支持组卷积，但是有`group`要与输入输出的通道数相等的约束。

<br/>

<font size=3>**Q: MindSpore支持矩阵转置吗？**</font>

A: 支持，请参考`mindspore.ops.Transpose`的[算子教程](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose)。

<br/>

<font size=3>**Q: 请问MindSpore能算给定任意一个`tensor`的方差吗？**</font>

A: mindspore目前暂无可以直接求出`tensor`方差的算子或接口。不过MindSpore有足够多的小算子可以支持用户实现这样的操作，你可以参考[class Moments(Cell)](https://www.mindspore.cn/docs/api/zh-CN/r1.6/_modules/mindspore/nn/layer/math.html#Moments)来实现。

<br/>

<font size=3>**Q: `nn.Embedding`层与PyTorch相比缺少了`Padding`操作，有其余的算子可以实现吗？**</font>

A: 在PyTorch中`padding_idx`的作用是将embedding矩阵中`padding_idx`位置的词向量置为0，并且反向传播时不会更新`padding_idx`位置的词向量。在MindSpore中，可以手动将embedding的`padding_idx`位置对应的权重初始化为0，并且在训练时通过`mask`的操作，过滤掉`padding_idx`位置对应的`Loss`。

<br/>

<font size=3>**Q: Operations中`Tile`算子执行到`__infer__`时`value`值为`None`，丢失了数值是怎么回事？**</font>

A: `Tile`算子的`multiples input`必须是一个常量（该值不能直接或间接来自于图的输入）。否则构图的时候会拿到一个`None`的数据，因为图的输入是在图执行的时候才传下去的，构图的时候拿不到图的输入数据。
相关的资料可以看[静态图语法支持](https://www.mindspore.cn/docs/note/zh-CN/r1.6/static_graph_syntax_support.html)。

<br/>

<font size=3>**Q: 使用conv2d算子将卷积核设置为(3,10),Tensor设置为[2,2,10,10]，在ModelArts上利用Ascend跑，报错: `FM_W+pad_left+pad_right-KW>=strideW`，CPU下不报错。**</font>

A: TBE(Tensor Boost Engine)算子是华为自研的Ascend算子开发工具，在TVM框架基础上扩展，进行自定义算子开发。上述问题是这个TBE算子的限制，x的width必须大于kernel的width。CPU的这个算子没有这个限制，所以不报错。

<br/>

<font size=3>**Q: 请问MindSpore实现了反池化操作了吗？类似于`nn.MaxUnpool2d` 这个反池化操作？**</font>

A: 目前 MindSpore 还没有反池化相关的接口。用户可以通过自定义算子的方式自行开发算子，详情请见[自定义算子](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/custom_operator.html)。

<br/>

<font size=3>**Q: 使用ExpandDims算子报错: `Pynative run op ExpandDims failed`。具体代码:**</font>

```python
context.set_context(mode=context.GRAPH_MODE,device_target='Ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=ops.ExpandDims()
output=expand_dims(input_tensor,0)
```

A: 这边的问题是选择了Graph模式却使用了PyNative的写法，所以导致报错，MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化:

- PyNative模式: 也称动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。

- Graph模式: 也称静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。

<br/>

<font size=3>**Q: Ascend后端报错：`AI CORE` 和`AI CPU`中都找不到有效的`kernel info`这个Kernel Select Failed时，如何定位？**</font>

A: Ascend后端，算子有AI CORE算子和AI CPU算子之分，部分算子AI CORE支持，部分算子AI CPU支持，部分算子两者同时支持。根据报错信息：

1. 如果`AI CORE`候选算子信息为空，则可能是在算子`check support`阶段，所有的算子信息均校验未通过。可以在日志中搜索关键字`CheckSupport`找到未通过的原因，根据具体信息修改shape或data type, 或者找开发人员进一步定位；
2. 如果`AI CPU`候选算子信息不为空，或者`AI CORE`和`AI CPU`候选算子信息都不为空，则可能是用户给到该算子的输入数据类型不在候选列表中，在选择阶段被过滤掉导致，可以根据候选列表尝试修改该算子的输入data type。

用户可以参考[官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/debug_in_pynative_mode.html)选择合适、统一的模式和写法来完成训练。
