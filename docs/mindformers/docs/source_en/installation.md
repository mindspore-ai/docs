# Installation Guidelines

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/installation.md)

## Confirming Version Matching Relationship

The currently supported hardware is the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server.

The current recommended Python version for the suite is 3.11.4.

|      MindSpore Transformers       |       MindSpore        |          CANN          |   Firmware & Drivers   |  Mirror Links  |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--------------:|
| In-Development Version | In-Development Version | In-Development Version | In-Development Version | Not applicable |

**Currently MindSpore Transformers recommends using a software package relationship as above.**

Historical version matching relationship:

|                     MindSpore Transformers                      |                  MindSpore                  |                                                                         CANN                                                                         |                                                                  Firmware & Drivers                                                                   |
|:----------------------------------------------------:|:-------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|
| [1.3.2](https://pypi.org/project/mindformers/1.3.2/) | [2.4.10](https://www.mindspore.cn/install/) | [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) | [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) |
| [1.2.0](https://pypi.org/project/mindformers/1.2.0/) | [2.3.0](https://www.mindspore.cn/install/)  |                     [8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)                     |                                       [24.1.RC2](https://www.hiascend.com/hardware/firmware-drivers/community)                                        |

## Installing Dependent Software

1. Install Firmware and Driver: Download the firmware and driver package through the [Confirming Version Matching Relationship](https://www.mindspore.cn/mindformers/docs/en/dev/installation.html#confirming-version-matching-relationship) to download the installation package, and refer to the [Ascend official tutorial](https://www.hiascend.com/en/document) for installation.

2. Install CANN and MindSpore: Follow the [Manual Installation](https://www.mindspore.cn/install/en) section on the MindSpore website for installation.

## Installing MindSpore Transformers

Currently only source code compilation installation is supported for in-development version, users can execute the following command to install MindSpore Transformers:

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## Installation Verification

To determine whether MindSpore Transformers has been successfully installed, execute the following code:

```bash
python -c "import mindformers as mf;mf.run_check()"
```

A similar result as below proves that the installation was successful:

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```