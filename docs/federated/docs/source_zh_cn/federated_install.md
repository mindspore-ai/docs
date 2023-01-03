# 获取MindSpore Federated

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/federated/docs/source_zh_cn/federated_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

[MindSpore Federated](https://gitee.com/mindspore/federated)框架代码现已独立建仓，分为端侧和云侧，其云侧能力依赖MindSpore和MindSpore Federated，利用MindSpore进行云侧集群聚合训练以及与端侧进行通信，因此需要分别获取MindSpore whl包和MindSpore Federated whl包。端侧能力依赖MindSpore Lite和MindSpore Federated java包，其中MindSpore Federated java主要负责数据预处理、调用MindSpore Lite进行模型训练和推理以及利用隐私保护机制和云侧进行模型相关的上传和下载。

## 获取MindSpore whl包

包括源码编译和下载发布版两种方式，支持x86 CPU、GPU CUDA等硬件平台，根据硬件平台类型，选择进行安装即可。安装步骤可参考[MindSpore安装指南](https://www.mindspore.cn/install)。

## 获取MindSpore Lite java包

包括源码编译和下载发布版两种方式。目前，MindSpore Lite联邦学习功能只支持Linux和Android平台，且只支持CPU。安装步骤可参考[下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r2.0.0-alpha/use/downloads.html)和[编译MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r2.0.0-alpha/use/build.html)。

## 获取MindSpore Federated whl包

包括源码编译和下载发布版两种方式。目前，MindSpore Federated联邦学习功能只支持Linux和Android平台。安装步骤可参考[编译MindSpore Federated whl](https://www.mindspore.cn/federated/docs/zh-CN/r2.0.0-alpha/deploy_federated_server.html)。

## 获取MindSpore Federated java包

包括源码编译和下载发布版两种方式。目前，MindSpore Federated联邦学习功能只支持Linux和Android平台。安装步骤可参考和[编译MindSpore Federated java](https://www.mindspore.cn/federated/docs/zh-CN/r2.0.0-alpha/deploy_federated_client.html)。

## Linux编译环境要求

目前源码编译只支持Linux，编译环境要求可参考[MindSpore源码编译](https://www.mindspore.cn/install)和[MindSpore Lite源码编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.0.0-alpha/use/build.html)。
