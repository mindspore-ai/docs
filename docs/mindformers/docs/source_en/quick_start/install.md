# Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/quick_start/install.md)

## Version Matching Relationship

The currently supported hardware is the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server.

The current recommended Python version for the suite is 3.10.

|      MindFormers       |       MindSpore        |          CANN          |   Firmware & Drivers   |  Mirror Links  |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--------------:|
| In-Development Version | In-Development Version | In-Development Version | In-Development Version | Not applicable |

**Currently MindFormers recommends using a software package relationship as above.**

Historical version matching relationship:

|                     MindFormers                      |                 MindSpore                  |                                                     CANN                                                     |                            Firmware & Drivers                            |                             Mirror Links                             |
|:----------------------------------------------------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| [1.3.0](https://pypi.org/project/mindformers/1.3.0/) | [2.4.0](https://www.mindspore.cn/install/) | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) | [24.1.RC3](https://www.hiascend.com/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |
| [1.2.0](https://pypi.org/project/mindformers/1.2.0/) | [2.3.0](https://www.mindspore.cn/install/) | [8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) | [24.1.RC2](https://www.hiascend.com/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

## Environment Installation

1. Install Firmware and Driver: Download the firmware and driver package through the [Version Matching Relationship](https://www.mindspore.cn/mindformers/docs/en/dev/quick_start/install.html#version-matching-relationship) to download the installation package, and refer to the [Ascend official tutorial](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html) for installation.

2. Install CANN and MindSpore: Use the officially provided Docker image (CANN, MindSpore are already included in the image, no need to install them manually) or follow the [Manual Installation](https://www.mindspore.cn/install/en#manual-installation) section on the MindSpore website for installation.

## MindFormers Installation

Currently only source code compilation installation is supported for in-development version, users can execute the following command to install MindFormers:

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
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```