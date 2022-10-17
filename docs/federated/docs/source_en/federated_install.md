# Obtaining MindSpore Federated

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/federated_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Currently, the [MindSpore Federated](https://gitee.com/mindspore/federated) framework code has been built independently, divided into device-side and cloud-side. Its cloud-side capability relies on MindSpore and MindSpore Federated, using MindSpore for cloud-side cluster aggregation training and communication with device-side, so it needs to get MindSpore whl package and MindSpore Federated whl package respectively. The device-side capability relies on MindSpore Lite and MindSpore Federated java packages, where MindSpore Federated java is mainly responsible for data pre-processing, model training and inference by calling MindSpore Lite for, as well as model-related uploads and downloads by using privacy protection mechanisms and the cloud side.

## Obtaining the MindSpore WHL Package

You can use the source code or download the release version to install MindSpore on hardware platforms such as the x86 CPU and GPU CUDA. For details about the installation process, see [Install](https://www.mindspore.cn/install/en) on the MindSpore website.

## Obtaining the MindSpore Lite Java Package

You can use the source code or download the release version. Currently, only the Linux and Android platforms are supported, and only the CPU hardware architecture is supported. For details about the installation process, see [Downloading MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) and [Building MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/build.html).

## Obtaining MindSpore Federated WHL Package

You can use the source code or download the release version. Currently, MindSpore Federated Learing supports the Linux and Android platforms. For details about the installation process, see [Building MindSpore Federated whl](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_server.html).

## Obtaining MindSpore Federated Java Package

You can use the source code or download the release version. Currently, MindSpore Federated Learing supports the Linux and Android platforms. For details about the installation process, see [Building MindSpore Federated java](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html).

## Requirements for Building the Linux Environment

Currently, the source code build is supported only in the Linux environment. For details about the environment requirements, see [MindSpore Source Code Build](https://www.mindspore.cn/install/en) and [MindSpore Lite Source Code Build](https://www.mindspore.cn/lite/docs/en/master/use/build.html).
