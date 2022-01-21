# 特性咨询

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/faq/source_zh_cn/feature_advice.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

<font size=3>**Q: 导出MindIR格式的时候，`input=np.random.uniform(...)`是不是固定格式？**</font>

A: 不是固定格式的，这一步操作是为了创建一个输入，以便于构建网络结构。`export`里只要传入正确的`shape`即可，使用`np.ones`和`np.zeros`创建都是可以的。

<br/>

<font size=3>**Q: MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？**</font>

A:  MindSpore采用Protobuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

<font size=3>**Q: 在使用ckpt或导出模型的过程中，报Protobuf内存限制错误，如何处理？**</font>

A: 当单条Protobuf数据过大时，因为Protobuf自身对数据流大小的限制，会报出内存限制的错误。这时可通过设置环境变量`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`解除限制。

<br/>

<font size=3>**Q: PyNative模式和Graph模式的区别？**</font>

A: 通过下面四个方面进行对比：

- 网络执行：两个模式使用的算子是一致的，因此相同的网络和算子，分别在两个模式下执行时，精度效果是一致的。由于Graph模式运用了图优化、计算图整图下沉等技术，Graph模式执行网络的性能和效率更高；

- 场景使用：Graph模式需要一开始就构建好网络结构，然后框架做整图优化和执行，比较适合网络固定没有变化，且需要高性能的场景；

- 不同硬件（`Ascend`、`GPU`和`CPU`）资源：都支持这两种模式；

- 代码调试：由于PyNative模式是逐行执行算子，用户可以直接调试Python代码，在代码中任意位置打断点查看对应算子的输出或执行结果。而Graph模式由于在构造函数里只是完成网络构造，实际没有执行，因此在`construct`函数里打断点无法获取对应算子的输出，只能先指定算子进行打印，然后在网络执行完成后查看输出结果。

<br/>

<font size=3>**Q: 使用MindSpore在GPU上训练的网络脚本可以不做修改直接在Ascend上进行训练么？**</font>

A: 可以的，MindSpore面向Ascend/GPU/CPU提供统一的API，在算子支持的前提下，网络脚本可以不做修改直接跨平台运行。

<br/>

<font size=3>**Q: 一个环境中如果既安装了MindSpore，又安装了PyTorch，是否在一个python文件中可以混用两个框架的语法呢？**</font>

A: 可以在一个python文件中混用两个框架的。要注意类型间的区别。例如两个框架创建的Tensor类型是不同的，但对于python的基础类型都是通用的。

<br/>

<font size=3>**Q: MindSpore可以读取TensorFlow的ckpt文件吗？**</font>

A: MindSpore的`ckpt`和TensorFlow的`ckpt`格式是不通用的，虽然都是使用`Protobuf`协议，但是`proto`的定义是不同的。当前MindSpore不支持读取TensorFlow或PyTorch的`ckpt`文件。

<br/>

<font size=3>**Q: 用MindSpore训练出的模型如何在Ascend 310上使用？可以转换成适用于HiLens Kit用的吗？**</font>

A: Ascend 310需要运行专用的OM模型,先使用MindSpore导出ONNX或AIR模型，再转化为Ascend 310支持的OM模型。具体可参考[多平台推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/multi_platform_inference_ascend_310.html)。可以，HiLens Kit是以Ascend 310为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型.

<br/>

<font size=3>**Q:MindSpore只能在华为自己的`Ascend`上跑么？**</font>

A: MindSpore同时支持华为自己的`Ascend`、`GPU`与`CPU`，是支持异构算力的。

<br/>

<font size=3>**Q: MindSpore在Ascend 310上是否可以转AIR模型？**</font>

A: Ascend 310不能导出AIR，需要在Ascend 910加载训练好的checkpoint后,导出AIR，然后在Ascend 310转成OM模型进行推理。Ascend 910的安装方法可以参考官网MindSpore[安装指南](https://www.mindspore.cn/install)。

<br/>

<font size=3>**Q: MindSpore对导出、导入模型的单个Tensor输入大小有什么限制？**</font>

A: 由于Protobuf的硬件限制，导出AIR、ONNX格式时，模型参数大小不能超过2G；导出MINDIR格式时，模型参数大小没有限制，MindSpore不支持导入AIR、ONNX格式，只支持MINDIR，导入大小的限制与导出一致。

<br/>

<font size=3>**Q: 安装运行MindSpore时，是否要求平台有GPU计算单元？需要什么硬件支持？**</font>

A: MindSpore当前支持CPU/GPU/Ascend。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来使用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/docs/note/zh-CN/r1.6/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/r1.6/RELEASE.md#)获取最新信息。

<br/>

<font size=3>**Q: 针对异构计算单元的支持，MindSpore有什么计划？**</font>

A: MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

<font size=3>**Q: MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？**</font>

A: ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

<font size=3>**Q: 最近出来的taichi编程语言有Python扩展，类似`import taichi as ti`就能直接用了，MindSpore是否也支持？**</font>

A: MindSpore支持Python原生表达，`import mindspore`相关包即可使用。

<br/>

<font size=3>**Q: 请问MindSpore支持梯度截断吗？**</font>

A: 支持，可以参考[梯度截断的定义和使用](https://gitee.com/mindspore/models/blob/master/official/nlp/transformer/src/transformer_for_train.py#L35)。

<br/>

<font size=3>**Q: MindSpore的IR设计理念是什么？**</font>

A: 函数式: 一切皆函数，易于微分实现；无副作用，易于实现自动并行化分析。`JIT`编译能力: 图形IR，控制流依赖和数据流合一，平衡通用性/易用性。图形完备的IR: 更多的转换`Python`灵活语法，包括递归等。

<br/>

<font size=3>**Q: MindSpore并行模型训练的优势和特色有哪些？**</font>

A: MindSpore分布式训练除了支持数据并行，还支持算子级模型并行，可以对算子输入tensor进行切分并行。在此基础上支持自动并行，用户只需要写单卡脚本，就能自动切分到多个节点并行执行。

<br/>

<font size=3>**Q: MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？**</font>

A: MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

<br/>

<font size=3>**Q: 当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？**</font>

A: MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

<br/>

<font size=3>**Q: MindSpore自动并行支持情况如何？**</font>

A: 自动并行特性对CPU GPU的支持还在完善中。推荐用户在Ascend 910 AI处理器上使用自动并行，可以关注开源社区，申请MindSpore开发者体验环境进行试用。

<br/>

<font size=3>**Q: MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？**</font>

A: TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。

<br/>

<font size=3>**Q: 使用PyNative模式能够进行迁移学习？**</font>

A: PyNative模式是兼容迁移学习的，更多的教程信息，可以参考[预训练模型加载代码详解](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/cv_mobilenetv2_fine_tune.html#id7)。

<br/>

<font size=3>**Q: MindSpore仓库中的[ModelZoo](https://gitee.com/mindspore/models/tree/master)和昇腾官网的[ModelZoo](https://www.hiascend.com/software/modelzoo)有什么关系？**</font>

A: MindSpore的ModelZoo主要提供MindSpore框架实现的模型，同时包括了Ascend/GPU/CPU/Mobile多种设备的支持。昇腾的ModelZoo主要提供运行于Ascend加速芯片上的模型，包括了MindSpore/PyTorch/TensorFlow/Caffe等多种框架的支持。可以参考对应的[Gitee仓库](https://gitee.com/ascend/modelzoo)

其中MindSpore+Ascend的组合是有重合的，这部分模型会以MindSpore的ModelZoo为主要版本，定期向昇腾ModelZoo发布。

<br/>

<font size=3>**Q: Ascend与NPU是什么关系？**</font>

A: NPU指针对神经网络算法的专用处理器，不同公司推出的NPU架构各异，Ascend是基于华为公司自研的达芬奇架构的NPU。
