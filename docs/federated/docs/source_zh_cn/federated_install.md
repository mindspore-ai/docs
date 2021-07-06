# 获取MindSpore Federated

`MindSpore Federated` `安装`

<!-- TOC -->

- [获取MindSpore Federated](#获取mindspore-federated)
    - [安装概述](#安装概述)
        - [获取MindSpore whl包](#获取mindspore-whl包)
        - [获取MindSpore Lite java包](#获取mindspore-lite-java包)
        - [Linux环境编译要求](#linux环境编译要求)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/federated_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 安装概述

  目前MindSpore Federated框架代码已经分别集成到云侧MindSpore和端侧Lite框架中，因此需要分别获取MindSpore whl包和MindSpore Lite java安装包。其中MindSpore Whl包负责云侧集群聚合训练以及和Lite通信。MindSpore Lite java包中包括两部分，一部分是MindSpore Lite训练安装包负责模型的底层训练，另一部分是Federated-Client安装包，负责模型的下发，加密以及和云侧MindSpore服务的交互。

### 获取MindSpore whl包

  包括源码和下载发布版两种方式，支持CPU、GPU等硬件平台，任选其一安装即可。安装流程可参考MindSpore安装指南[安装章节](https://www.mindspore.cn/install)。

### 获取MindSpore Lite java包

  包括源码和下载发布版两种方式。目前只支持x86和android平台，只支持CPU硬件架构。安装流程可参考MindSpore Lite教程[下载章节](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)和[编译章节](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)。详见下文部署Federated-Client章节。

### Linux环境编译要求

  目前源码编译只支持linux环境，环境要求可参考，[MindSpore源码编译](https://www.mindspore.cn/install) 和[MindSpore Lite源码编译](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html)。
