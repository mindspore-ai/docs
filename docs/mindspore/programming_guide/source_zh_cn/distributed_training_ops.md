# 集合通信原语

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [集合通信原语](#集合通信原语)
    - [AllReduce](#AllReduce)
    - [AllGather](#AllGather)
    - [ReduceScatter](#ReduceScatter)
    - [BroadCast](#BroadCast)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/distributed_training_ops.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

在分布式训练中涉及通信操作例如`AllReduce`、`ReduceScatter`、`AllGather`和`BroadCast`等操作，如下图所示：

![image](./images/communication.png)

## AllReduce

`AllReduce`操作会将每卡对应输入`tensor`进行求和操作，最终每张卡输出是相同的`tensor`，例如上图左上部分所示，每张卡各自的输入为`0, 1, 2, 3`，经过`AllReduce`之后，每张卡输出的结果为每张卡输入之和为6(0+1+2+3)。

## AllGather

`AllGather`操作会将每张卡的输入在第0维度上进行拼接，最终每张卡输出是相同的`tensor`。例如上图右上部分所示，每卡的输入是大小为1x1的`tensor`，经过`AllGather`操作之后，每卡的输入都构成了输出的一部分，对应的输出shape为[4,1]。其中索引为[0,0]元素值来自于`rank0`的输入，索引为[1,0]的元素值来自于`rank1`的输入。

## ReduceScatter

`ReduceScatter`操作会将每张卡的输入先进行求和(`Reduce`)，然后在第0为维度按卡数切分，分发到对应的卡上。例如上图右下角所示，每卡的输入均为4x1的`tensor`，先进行求和得到[0,4, 8, 12]的`tensor`，然后进行分发，每卡获得1x1大小的`tensor`。

## BroadCast

`BroadCast`操作是将某张卡的输入广播到其他卡上，常见于参数的初始化。例如将0卡大小为1x1的`tensor`进行广播，最终每张卡的结果均为相同的[[0]]。
