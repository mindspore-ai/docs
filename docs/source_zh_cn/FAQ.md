# FAQ

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `模型导出` `模型训练` `初级` `中级` `高级`

<!-- TOC -->

- [FAQ](#faq)
    - [安装类](#安装类)
        - [pip安装](#pip安装)
        - [源码编译安装](#源码编译安装)
        - [环境变量](#环境变量)
        - [安装验证](#安装验证)
    - [算子支持](#算子支持)
    - [网络模型](#网络模型)
    - [平台系统](#平台系统)
    - [后端运行](#后端运行)
    - [编程语言拓展](#编程语言拓展)
    - [特性支持](#特性支持)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/source_zh_cn/FAQ.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 安装类

### pip安装

Q：安装MindSpore版本：GPU、CUDA 10.1、0.5.0-beta、Ubuntu-x86，出现问题：`cannot open shared object file:file such file or directory`。

A：从报错情况来看，是cublas库没有找到。一般的情况下是cublas库没有安装，或者是因为没有加入到环境变量中去。通常cublas是随着cuda以及驱动一起安装的，确认安装后把cublas所在的目录加入`LD_LIBRARY_PATH`环境变量中即可。

<br/>

Q：使用pip安装时报错：`SSL:CERTIFICATE_VERIFY_FATLED`应该怎么办？

A：在pip安装命令后添加参数 `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com`重试即可。

<br/>

Q：pip安装MindSpore对Python版本是否有特别要求？

A：MindSpore开发过程中用到了Python3.7+的新特性，因此建议您通过`conda`工具添加Python3.7.5的开发环境。

<br/>

Q：使用pip安装时报错`ProxyError(Cannot connect to proxy)`，应该怎么办？

A：此问题一般是代理配置问题，Ubuntu环境下可通过`export http_proxy={your_proxy}`设置代理；Windows环境可以在cmd中通过`set http_proxy={your_proxy}`进行代理设置。

<br/>

Q：使用pip安装时提示错误，应该怎么办？

A：请执行`pip -V`查看是否绑定了Python3.7+。如果绑定的版本不对，建议使用`python3.7 -m pip install`代替`pip install`命令。

<br/>

Q：MindSpore网站安装页面找不到MindInsight和MindArmour的whl包，无法安装怎么办？

A：您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

<br/>

### 源码编译安装

Q：MindSpore安装：版本0.6.0-beta + Ascend 910 + Ubuntu_aarch64 + Python3.7.5，手动下载对应版本的whl包，编译并安装gmp6.1.2。其他Python库依赖已经安装完成，执行样例失败，报错显示找不到so文件。

A：`libdatatransfer.so`动态库是`fwkacllib/lib64`目录下的，请先在`/usr/local`目录find到这个库所在的路径，然后把这个路径加到`LD_LIBRARY_PATH`环境变量中，确认设置生效后，再执行。

<br/>

Q：源码编译MindSpore过程时间过长，或时常中断该怎么办？

A：MindSpore通过submodule机制引入第三方依赖包，其中`protobuf`依赖包（v3.8.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

Q：如何改变第三方依赖库安装路径？

A：第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

Q：MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？

A：当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**：MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）。

<br/>

Q：当源码编译MindSpore，提示`tclsh not found`时，应该怎么办？

A：当有此提示时说明要用户安装`tclsh`；如果仍提示缺少其他软件，同样需要安装其他软件。

<br/>

### 环境变量

Q: 一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？

A: 常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

### 安装验证

Q:个人电脑CPU环境安装MindSpore后验证代码时报错：`the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？
```python
import numpy as np
from mindspore import Tensor
from mindspore.ops import functional as F
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(F.tensor_add(x,y))
```

A：CPU硬件平台安装MindSpore后测试是否安装成功,只需要执行命令：`python -c 'import mindspore'`，如果没有显示`No module named 'mindspore'`等错误即安装成功。问题中的验证代码仅用于验证Ascend平台安装是否成功。

<br/>

## 算子支持

Q：官网的LSTM示例在Ascend上跑不通

A：目前LSTM只支持在GPU和CPU上运行，暂不支持硬件环境，您可以[点击这里](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html)查看算子支持情况。

<br/>

Q：conv2d设置为(3,10),Tensor[2,2,10,10]，在ModelArts上利用Ascend跑，报错：`FM_W+pad_left+pad_right-KW>=strideW`，CPU下不报错。

A：这是TBE这个算子的限制，x的width必须大于kernel的width。CPU的这个算子没有这个限制，所以不报错。

<br/>

## 网络模型

Q：MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？

A： MindSpore采用protbuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

Q：用MindSpore训练出的模型如何在Ascend 310上使用？可以转换成适用于HiLens Kit用的吗？

A：Ascend 310需要运行专用的OM模型,先使用MindSpore导出ONNX或AIR模型，再转化为Ascend 310支持的OM模型。具体可参考[多平台推理](https://www.mindspore.cn/tutorial/zh-CN/master/use/multi_platform_inference.html)。可以，HiLens Kit是以Ascend 310为推理核心，所以前后两个问题本质上是一样的，需要转换为OM模型.

<br/>

Q：MindSpore如何进行参数（如dropout值）修改？

A：在构造网络的时候可以通过 `if self.training: x = dropput(x)`，验证的时候，执行前设置`network.set_train(mode_false)`，就可以不适用dropout，训练时设置为True就可以使用dropout。

<br/>

Q：从哪里可以查看MindSpore训练及推理的样例代码或者教程？

A：可以访问[MindSpore官网教程](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)。

<br/>

Q：MindSpore支持哪些模型的训练？

A：MindSpore针对典型场景均有模型训练支持，支持情况详见[Release note](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md)。

<br/>

Q：MindSpore有哪些现成的推荐类或生成类网络或模型可用？

A：目前正在开发Wide & Deep、DeepFM、NCF等推荐类模型，NLP领域已经支持Bert_NEZHA，正在开发MASS等模型，用户可根据场景需要改造为生成类网络，可以关注[MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

<br/>

Q：MindSpore模型训练代码能有多简单？

A：除去网络定义，MindSpore提供了Model类的接口，大多数场景只需几行代码就可完成模型训练。

<br/>

## 平台系统

Q：Ascend 310 不能安装MindSpore么？

A：Ascend 310只能用作推理，MindSpore支持在Ascend 910训练，训练出的模型转化为OM模型可用于Ascend 310上推理。

<br/>

Q：安装运行MindSpore时，是否要求平台有GPU、NPU等计算单元？需要什么硬件支持？

A：MindSpore当前支持CPU/GPU/Ascend /NPU。目前笔记本电脑或者有GPU的环境，都可以通过Docker镜像来试用。当前MindSpore Model Zoo中有部分模型已经支持GPU的训练和推理，其他模型也在不断地进行完善。在分布式并行训练方面，MindSpore当前支持GPU多卡训练。你可以通过[RoadMap](https://www.mindspore.cn/docs/zh-CN/master/roadmap.html)和项目[Release note](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md)获取最新信息。

<br/>

Q：针对异构计算单元的支持，MindSpore有什么计划？

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

<br/>

Q：MindSpore与ModelArts是什么关系，在ModelArts中能使用MindSpore吗？

A：ModelArts是华为公有云线上训练及推理平台，MindSpore是华为深度学习框架，可以查阅[MindSpore官网教程](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/use_on_the_cloud.html)，教程中详细展示了用户如何使用ModelArts来做MindSpore的模型训练。

<br/>

Q：MindSpore是否支持Windows 10？

A：MindSpore CPU版本已经支持在Windows 10系统中安装，具体安装步骤可以查阅[MindSpore官网教程](https://www.mindspore.cn/install/)。

<br/>

## 后端运行

Q：请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？

A：首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

Q：MindSpore的operation算子报错：`device target [CPU] is not supported in pynative mode`

A：pynative 模式目前只支持Ascend和GPU，暂时还不支持CPU。

<br/>

Q：在Ascend平台上，执行用例有时候会报错run time error，如何获取更详细的日志帮助问题定位？

A：可以通过开启slog获取更详细的日志信息以便于问题定位，修改`/var/log/npu/conf/slog/slog.conf`中的配置，可以控制不同的日志级别，对应关系为：0:debug、1:info、2:warning、3:error、4:null(no output log)，默认值为1。

<br/>

Q：使用ExpandDims算子报错：`Pynative run op ExpandDims failed`。具体代码：

```python
context.set_context(
mode=cintext.GRAPH_MODE,
device_target='ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=P.ExpandDims()
output=expand_dims(input_tensor,0)
```

A：这边的问题是选择了Graph模式却使用了PyNative的写法，所以导致报错，MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化:

- PyNative模式：也称动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。

- Graph模式：也称静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。
用户可以参考[官网教程](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/debugging_in_pynative_mode.html)选择合适、统一的模式和写法来完成训练。

<br/>

## 编程语言拓展

Q：最近出来的taichi编程语言有Python扩展，类似`import taichi as ti`就能直接用了，MindSpore是否也支持？

A：MindSpore支持Python原生表达，`import mindspore`相关包即可使用。

<br/>

Q：MindSpore是否（计划）支持多语言扩展？

A：MindSpore目前支持Python扩展，针对C++、Rust、Julia等语言的支持正在开发中。

<br/>

## 特性支持

Q：MindSpore并行模型训练的优势和特色有哪些？

A：MindSpore分布式训练除了支持数据并行，还支持算子级模型并行，可以对算子输入tensor进行切分并行。在此基础上支持自动并行，用户只需要写单卡脚本，就能自动切分到多个节点并行执行。

<br/>

Q：请问MindSpore实现了反池化操作了吗？类似于`nn.MaxUnpool2d` 这个反池化操作？

A：目前 MindSpore 还没有反池化相关的接口。如果用户想自己实现的话，可以通过自定义算子的方式自行开发算子,自定义算子[详见这里](https://www.mindspore.cn/tutorial/zh-CN/master/use/custom_operator.html)。

<br/>

Q：MindSpore有轻量的端侧推理引擎么？

A：MindSpore轻量化推理框架MindSpore Lite已于r0.7版本正式上线，欢迎试用并提出宝贵意见，概述、教程和文档等请参考[MindSpore Lite](https://www.mindspore.cn/lite)

<br/>

Q：MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？

A：MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

<br/>

Q：当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？

A：MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

<br/>

Q：MindSpore自动并行支持情况如何？

A：自动并行特性对CPU GPU的支持还在完善中。推荐用户在Ascend 910 AI处理器上使用自动并行，可以关注开源社区，申请MindSpore开发者体验环境进行试用。

<br/>

Q：MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？

A：TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。

<br/>

Q：其他框架的脚本或者模型怎么迁移到MindSpore？

A：关于脚本或者模型迁移，可以查询MindSpore官网中关于[网络迁移](https://www.mindspore.cn/tutorial/zh-CN/master/advanced_use/network_migration.html)的介绍。

<br/>

Q：MindSpore是否附带开源电商类数据集？

A：暂时还没有，可以持续关注[MindSpore官网](https://www.mindspore.cn)。