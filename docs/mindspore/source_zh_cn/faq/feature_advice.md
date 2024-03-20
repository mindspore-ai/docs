# 特性咨询

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/faq/feature_advice.md)

## Q: 导出MindIR格式的时候，`input=np.random.uniform(...)`是不是固定格式？

A: 不是固定格式的，这一步操作是为了创建一个输入，以便于构建网络结构。`export`里只要传入正确的`shape`即可，使用`np.ones`和`np.zeros`创建都是可以的。

<br/>

## Q: MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？

A:  MindSpore采用Protobuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

## Q: PyNative模式和Graph模式的区别？

A: 通过下面四个方面进行对比：

- 网络执行：两个模式使用的算子是一致的，因此相同的网络和算子，分别在两个模式下执行时，精度效果是一致的。由于Graph模式运用了图优化、计算图整图下沉等技术，Graph模式执行网络的性能和效率更高；

- 场景使用：Graph模式需要一开始就构建好网络结构，然后框架做整图优化和执行，比较适合网络固定没有变化，且需要高性能的场景；

- 不同硬件（`Ascend`、`GPU`和`CPU`）资源：都支持这两种模式；

- 代码调试：由于PyNative模式是逐行执行算子，用户可以直接调试Python代码，在代码中任意位置打断点查看对应算子的输出或执行结果。而Graph模式由于在构造函数里只是完成网络构造，实际没有执行，因此在`construct`函数里打断点无法获取对应算子的输出，只能先指定算子进行打印，然后在网络执行完成后查看输出结果。

- 语法支持：PyNative模式具有动态语法亲和性，表达灵活，基本覆盖所有Python语法。Graph模式支持覆盖Python常用语法子集，以支持神经网络的构建和训练。

<br/>

## Q: 使用MindSpore在GPU上训练的网络脚本可以不做修改直接在Ascend上进行训练么？

A: 可以的，MindSpore面向Ascend/GPU/CPU提供统一的API，在算子支持的前提下，网络脚本可以不做修改直接跨平台运行。

<br/>

## Q: 一个环境中如果既安装了MindSpore，又安装了PyTorch，是否在一个python文件中可以混用两个框架的语法呢？

A: 可以在一个python文件中混用两个框架的。要注意类型间的区别。例如两个框架创建的Tensor类型是不同的，但对于python的基础类型都是通用的。

<br/>

## Q: MindSpore可以读取TensorFlow的ckpt文件吗？

A: MindSpore的`ckpt`和TensorFlow的`ckpt`格式是不通用的，虽然都是使用`Protobuf`协议，但是`proto`的定义是不同的。当前MindSpore不支持读取TensorFlow或PyTorch的`ckpt`文件。

<br/>

## Q: 用MindSpore训练出的模型如何在Atlas 200/300/500推理产品上使用？可以转换成适用于HiLens Kit用的吗？

A: Atlas 200/300/500推理产品需要运行专用的OM模型，先使用MindSpore导出ONNX或AIR模型，再转化为Atlas 200/300/500推理产品支持的OM模型。可以，HiLens Kit是以Atlas 200/300/500推理产品为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型。

<br/>

## Q: MindSpore只能在华为自己的`Ascend`上跑么？

A: MindSpore同时支持华为自己的`Ascend`、`GPU`与`CPU`，是支持异构算力的。

<br/>

## Q: MindSpore在Atlas 200/300/500推理产品上是否可以转AIR模型？

A: Atlas 200/300/500推理产品不能导出AIR，需要在Atlas训练系列产品加载训练好的checkpoint后，导出AIR，然后在Atlas 200/300/500推理产品转成OM模型进行推理。Atlas训练系列产品的安装方法可以参考官网MindSpore[安装指南](https://www.mindspore.cn/install)。

<br/>

## Q: MindSpore对导出、导入模型的单个Tensor输入大小有什么限制？

A: 由于Protobuf的硬件限制，导出AIR、ONNX格式时，模型参数大小不能超过2G；导出MINDIR格式时，模型参数大小没有限制，MindSpore不支持导入AIR、ONNX格式，只支持导入MINDIR。 MINDIR的导入不存在模型参数大小限制。

<br/>

## Q: 安装运行MindSpore时，是否要求平台有GPU计算单元？需要什么硬件支持？

A: MindSpore当前支持CPU/GPU/Ascend。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来使用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过项目[Release note](https://gitee.com/mindspore/mindspore/blob/r2.3/RELEASE.md#)获取最新信息。

<br/>

## Q: 针对异构计算单元的支持，MindSpore有什么计划？

A: MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

## Q: MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？

A: ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架。

<br/>

## Q: 最近出来的taichi编程语言有Python扩展，类似`import taichi as ti`就能直接用了，MindSpore是否也支持？

A: MindSpore支持Python原生表达，`import mindspore`相关包即可使用。

<br/>

## Q: 请问MindSpore支持梯度截断吗？

A: 支持，可以参考代码[梯度截断脚本](https://gitee.com/mindspore/models/blob/master/official/nlp/Transformer/src/transformer_for_train.py#L35)。

<br/>

## Q: MindSpore的IR设计理念是什么？

A: 函数式: 一切皆函数，易于微分实现；无副作用，易于实现自动并行化分析。`JIT`编译能力: 图形IR，控制流依赖和数据流合一，平衡通用性/易用性。图形完备的IR: 更多的转换`Python`灵活语法，包括递归等。

<br/>

## Q: MindSpore并行模型训练的优势和特色有哪些？

A: MindSpore分布式训练除了支持数据并行，还支持算子级模型并行，可以对算子输入tensor进行切分并行。在此基础上支持自动并行，用户只需要写单卡脚本，就能自动切分到多个节点并行执行。

<br/>

## Q: MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？

A: MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

<br/>

## Q: 当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？

A: MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

<br/>

## Q: MindSpore自动并行支持情况如何？

A: 自动并行特性对CPU GPU的支持还在完善中。推荐用户在Atlas训练系列产品上使用自动并行，可以关注开源社区，申请MindSpore开发者体验环境进行试用。

<br/>

## Q: MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？

A: TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。

<br/>

## Q: 使用PyNative模式能够进行迁移学习？

A: PyNative模式是兼容迁移学习的。

<br/>

## Q: MindSpore仓库中的[ModelZoo](https://gitee.com/mindspore/models/blob/master/README_CN.md#)和昇腾官网的[ModelZoo](https://www.hiascend.com/software/modelzoo)有什么关系？

A: MindSpore的ModelZoo主要提供MindSpore框架实现的模型，同时包括了Ascend/GPU/CPU/Mobile多种设备的支持。昇腾的ModelZoo主要提供运行于Ascend加速芯片上的模型，包括了MindSpore/PyTorch/TensorFlow/Caffe等多种框架的支持。可以参考对应的[Gitee仓库](https://gitee.com/ascend/modelzoo)。

其中MindSpore+Ascend的组合是有重合的，这部分模型会以MindSpore的ModelZoo为主要版本，定期向昇腾ModelZoo发布。

<br/>

## Q: Ascend与NPU是什么关系？

A: NPU指针对神经网络算法的专用处理器，不同公司推出的NPU架构各异，Ascend是基于华为公司自研的达芬奇架构的NPU。
