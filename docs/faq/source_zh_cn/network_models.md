# 网络模型类

`数据处理` `环境准备` `模型导出` `模型训练` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/network_models.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q：如何不将数据处理为MindRecord格式，直接进行训练呢？

A：可以使用自定义的数据加载方式 `GeneratorDataset`，具体可以参考[数据集加载](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html)文档中的自定义数据集加载。

<br/>

Q：MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？

A： MindSpore采用protbuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

Q：用MindSpore训练出的模型如何在Ascend 310上使用？可以转换成适用于HiLens Kit用的吗？

A：Ascend 310需要运行专用的OM模型,先使用MindSpore导出ONNX或AIR模型，再转化为Ascend 310支持的OM模型。具体可参考[多平台推理](https://www.mindspore.cn/tutorial/inference/zh-CN/master/multi_platform_inference_ascend_310.html)。可以，HiLens Kit是以Ascend 310为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型.

<br/>

Q：MindSpore如何进行参数（如dropout值）修改？

A：在构造网络的时候可以通过 `if self.training: x = dropput(x)`，验证的时候，执行前设置`network.set_train(mode_false)`，就可以不适用dropout，训练时设置为True就可以使用dropout。

<br/>

Q：从哪里可以查看MindSpore训练及推理的样例代码或者教程？

A：可以访问[MindSpore官网教程训练](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)和[MindSpore官网教程推理](https://www.mindspore.cn/tutorial/inference/zh-CN/master/index.html)。

<br/>

Q：MindSpore支持哪些模型的训练？

A：MindSpore针对典型场景均有模型训练支持，支持情况详见[Release note](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md#)。

<br/>

Q：MindSpore有哪些现成的推荐类或生成类网络或模型可用？

A：目前正在开发Wide & Deep、DeepFM、NCF等推荐类模型，NLP领域已经支持Bert_NEZHA，正在开发MASS等模型，用户可根据场景需要改造为生成类网络，可以关注[MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

<br/>

Q：MindSpore模型训练代码能有多简单？

A：除去网络定义，MindSpore提供了Model类的接口，大多数场景只需几行代码就可完成模型训练。
