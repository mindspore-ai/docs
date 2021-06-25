# 特性咨询

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/feature_advice.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：一个环境中如果既安装了MindSpore，又安装了PyTorch，是否在一个python文件中可以混用两个框架的语法呢？**</font>

A：可以在一个python文件中混用两个两个框架的。要注意类型间的区别。例如两个框架创建的Tensor类型是不同的，但对于python的基础类型都是通用的。

<br/>

<font size=3>**Q：MindSpore可以读取TensorFlow的ckpt文件吗？**</font>

A：MindSpore的`ckpt`和TensorFlow的`ckpt`格式是不通用的，虽然都是使用`protobuf`协议，但是`proto`的定义是不同的。当前MindSpore不支持读取TensorFlow或PyTorch的`ckpt`文件。

<br/>

<font size=3>**Q：用MindSpore训练出的模型如何在Ascend 310上使用？可以转换成适用于HiLens Kit用的吗？**</font>

A：Ascend 310需要运行专用的OM模型,先使用MindSpore导出ONNX或AIR模型，再转化为Ascend 310支持的OM模型。具体可参考[多平台推理](https://www.mindspore.cn/tutorial/inference/zh-CN/master/multi_platform_inference_ascend_310.html)。可以，HiLens Kit是以Ascend 310为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型.

<br/>

<font size=3>**Q:MindSpore只能在华为自己的`NPU`上跑么？**</font>

A: MindSpore同时支持华为自己的`Ascend NPU`、`GPU`与`CPU`，是支持异构算力的。

<br/>

<font size=3>**Q：MindSpore在Ascend 310上是否可以转AIR模型？**</font>

A：Ascend 310不能导出AIR，需要在Ascend 910加载训练好的checkpoint后,导出AIR，然后在Ascend 310转成OM模型进行推理。Ascend 910的安装方法可以参考官网MindSpore[安装指南](https://www.mindspore.cn/install)。

<br/>

<font size=3>**Q：MindSpore对导出、导入模型的单个Tensor输入大小有什么限制？**</font>

A：由于ProtoBuf的硬件限制，导出AIR、ONNX模型时，单个Tensor大小不能超过2G。导入的MindIR模型中，单个Tensor不能超过2G。

<br/>

<font size=3>**Q：安装运行MindSpore时，是否要求平台有GPU、NPU等计算单元？需要什么硬件支持？**</font>

A：MindSpore当前支持CPU/GPU/Ascend /NPU。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来试用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/doc/note/zh-CN/master/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md#)获取最新信息。

<br/>

<font size=3>**Q：针对异构计算单元的支持，MindSpore有什么计划？**</font>

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

<font size=3>**Q：MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？**</font>

A：ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

<font size=3>**Q：最近出来的taichi编程语言有Python扩展，类似`import taichi as ti`就能直接用了，MindSpore是否也支持？**</font>

A：MindSpore支持Python原生表达，`import mindspore`相关包即可使用。

<br/>

<font size=3>**Q：请问MindSpore支持梯度截断吗？**</font>

A：支持，可以参考[梯度截断的定义和使用](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/transformer/src/transformer_for_train.py#L35)。

<br/>

<font size=3>**Q：MindSpore的IR设计理念是什么？**</font>

A：函数式：一切皆函数，易于微分实现；无副作用，易于实现自动并行化分析；`JIT`编译能力：图形IR，控制流依赖和数据流合一，平衡通用性/易用性；图灵完备的IR：更多的转换`Python`灵活语法，包括递归等。

<br/>

<font size=3>**Q：MindSpore并行模型训练的优势和特色有哪些？**</font>

A：MindSpore分布式训练除了支持数据并行，还支持算子级模型并行，可以对算子输入tensor进行切分并行。在此基础上支持自动并行，用户只需要写单卡脚本，就能自动切分到多个节点并行执行。

<br/>

<font size=3>**Q：MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？**</font>

A：MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

<br/>

<font size=3>**Q：当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？**</font>

A：MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

<br/>

<font size=3>**Q：MindSpore自动并行支持情况如何？**</font>

A：自动并行特性对CPU GPU的支持还在完善中。推荐用户在Ascend 910 AI处理器上使用自动并行，可以关注开源社区，申请MindSpore开发者体验环境进行试用。

<br/>

<font size=3>**Q：MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？**</font>

A：TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。

<br/>

<font size=3>**Q：MindSpore Serving是否支持热更新，避免推理服务中断？**</font>

A：MindSpore Serving当前不支持热更新，需要用户重启；当前建议跑多个Serving服务，升级模型版本时，重启部分服务以避免服务中断。

<br/>

<font size=3>**Q：MindSpore Serving是否支持一个模型启动多个Worker，以支持多卡单模型并发？**</font>

A：MindSpore Serving暂未支持分流，即不支持一个模型启动多个Worker，这个功能正在开发中；当前建议跑多个Serving服务，通过对接多个Serving服务的服务器进行分流和负载均衡。另外，为了避免`master`和`worker`之间的消息转发，可以使用接口`start_servable_in_master`使`master`和`worker`执行在同一进程，实现Serving服务轻量级部署。

<br/>

<font size=3>**Q：MindSpore Serving的版本和MindSpore的版本如何配套？**</font>

A：MindSpore Serving配套相同版本号的MindSpore的版本，比如Serving `1.1.1`版本配套 MindSpore `1.1.1`版本。

<br/>

<font size=3>**Q:使用PyNative模式能够进行迁移学习？**</font>

A: PyNative模式是兼容迁移学习的，更多的教程信息，可以参考[预训练模型加载代码详解](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/cv_mobilenetv2_fine_tune.html#id7)。

<br/>

<font size=3>**Q:MindSpore仓库中的[ModelZoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)和昇腾官网的[ModelZoo](https://www.hiascend.com/software/modelzoo)有什么关系？**</font>

A: MindSpore的ModelZoo主要提供MindSpore框架实现的模型，同时包括了Ascend/GPU/CPU/Mobile多种设备的支持。昇腾的ModelZoo主要提供运行于Ascend加速芯片上的模型，包括了MindSpore/PyTorch/TensorFlow/Caffe等多种框架的支持。可以参考对应的[Gitee仓库](https://gitee.com/ascend/modelzoo)

其中MindSpore+Ascend的组合是有重合的，这部分模型会以MindSpore的ModelZoo为主要版本，定期向昇腾ModelZoo发布。
