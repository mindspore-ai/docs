# 获取MindSpore Federated

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/federated_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore Federated框架代码分别集成在云侧MindSpore和端侧Lite框架中，因此需要分别获取MindSpore whl包和MindSpore Lite java安装包。其中，MindSpore Whl包负责云侧集群聚合训练以及和Lite的通信。MindSpore Lite java包中包括两部分，一部分是MindSpore Lite训练安装包，负责模型的底层训练，另一部分是Federated-Client安装包，负责模型的下发、加密以及和云侧MindSpore服务的交互。

## 获取MindSpore whl包

包括源码和下载发布版两种方式，支持x86的CPU、GPU硬件平台，根据硬件平台选择安装即可。MindSpore从1.3.0版本开始支持联邦学习。安装步骤可参考[MindSpore安装指南](https://www.mindspore.cn/install)。

## 获取MindSpore Lite java包

包括源码和下载发布版两种方式。目前只支持x86和Android平台，只支持CPU硬件架构。MindSpore Lite从1.3.0版本开始支持联邦学习。安装步骤可参考[下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)和[编译MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)。

## Linux环境编译要求

目前源码编译只支持Linux环境，环境要求可参考[MindSpore源码编译](https://www.mindspore.cn/install)和[MindSpore Lite源码编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)。
