# Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/quick_start/install.md)

## Version Matching Relationship

The currently supported hardware is the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server.

The current recommended Python version for the suite is 3.9.

| MindFormers |                 MindSpore                  |                                                                                                                                                                                                                  CANN                                                                                                                                                                                                                   |                                                   Firmware & Drivers                                                    |                                 Mirror Links                                  |
|:-----------:|:------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
|   r1.3.0    | [2.3.0](https://www.mindspore.cn/install/) | 8.0.RC2.beta1 <br/> toolkit：[aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run) [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-x86_64.run) <br/> kernels：[kernels](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-kernels-910b_8.0.RC2_linux.run) | firmware：[firmware](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC2/Ascend-hdk-910b-npu-firmware_7.3.0.1.231.run) <br/> driver： [driver](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC2/Ascend-hdk-910b-npu-driver_24.1.rc2_linux-aarch64.run) | [image](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

**Currently MindFormers only supports software packages with relationships as above.**

## Environment Installation

1. Install Firmware and Driver: Download the firmware and driver package through the [Version Matching Relationship](https://www.mindspore.cn/mindformers/docs/en/dev/quick_start/install.html#version-matching-relationship) to download the installation package, and refer to the [Ascend official tutorial](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html) for installation.

2. Install CANN and MindSpore: Use the officially provided Docker image (CANN, MindSpore are already included in the image, no need to install them manually) or follow the [Manual Installation](https://www.mindspore.cn/install/en#manual-installation) section on the MindSpore website for installation.

## MindFormers Installation

Currently only source code compilation installation is supported, users can execute the following command to install MindFormers:

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## Installation Verification

To determine whether MindFormers has been successfully installed, execute the following command:

```python
import mindformers as mf
mf.run_check()
```

A similar result as below proves that the installation was successful:

```text
- INFO - MindFormers version: 1.2.0
- INFO - MindSpore version: 2.3.0
- INFO - Ascend-cann-toolkit version: 8.0.RC2
- INFO - Ascend driver version: 24.1.rc2
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```