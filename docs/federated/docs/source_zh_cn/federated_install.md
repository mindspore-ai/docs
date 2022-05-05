# 获取MindSpore Federated

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_zh_cn/federated_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

MindSpore Federated框架代码集成在云侧MindSpore和端侧MindSpore Lite框架中，因此需要分别获取MindSpore whl包和MindSpore Lite java安装包。其中，MindSpore Whl包负责云侧集群聚合训练，以及与Lite的通信。MindSpore Lite java安装包中包括两部分，一部分是MindSpore Lite训练安装包，负责模型的端侧本地训练，另一部分是Federated-Client安装包，负责模型的下发、加密以及与云侧MindSpore服务的交互。

## 获取MindSpore whl包

包括源码编译和下载发布版两种方式，支持x86 CPU、GPU CUDA等硬件平台，根据硬件平台类型，选择进行安装即可。MindSpore从1.3.0版本开始支持联邦学习。安装步骤可参考[MindSpore安装指南](https://www.mindspore.cn/install)。

## 获取MindSpore Lite java包

包括源码和下载发布版两种方式。MindSpore Lite从1.3.0版本开始支持联邦学习。目前，MindSpore Lite联邦学习功能只支持Linux和Android平台，且只支持CPU。安装步骤可参考[下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/downloads.html)和[编译MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/build.html)。

## Linux编译环境要求

目前源码编译只支持Linux，编译环境要求可参考[MindSpore源码编译](https://www.mindspore.cn/install)和[MindSpore Lite源码编译](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/build.html)。
