# Obtaining MindSpore Federated

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/federated_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Installation Overview

Currently, the MindSpore Federated framework code has been integrated into the MindSpore framework on the cloud and the Lite framework on the device. Therefore, you need to obtain the MindSpore WHL package and MindSpore Lite Java installation package separately. The MindSpore WHL package is used for cluster aggregation training on the cloud and communication with Lite. The MindSpore Lite Java package contains two parts. One is the MindSpore Lite training installation package, which is used for bottom-layer model training. The other is the Federated-Client installation package, which is used for model delivery, encryption, and interaction with the MindSpore service on the cloud.

### Obtaining the MindSpore WHL Package

You can use the source code or download the release version to install MindSpore on hardware platforms such as the x86 CPU and GPU. For details about the installation process, see [Install](https://www.mindspore.cn/install/en) on the MindSpore website.

### Obtaining the MindSpore Lite Java Package

You can use the source code or download the release version. Currently, only the x86 and Android platforms are supported, and only the CPU hardware architecture is supported. For details about the installation process, see [Downloading MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) and [Building MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/build.html). For details, see "Deploying Federated-Client."

### Requirements for Building the Linux Environment

Currently, the source code build is supported only in the Linux environment. For details about the environment requirements, see [MindSpore Source Code Build](https://www.mindspore.cn/install/en) and [MindSpore Lite Source Code Build](https://www.mindspore.cn/lite/docs/en/master/use/build.html).
