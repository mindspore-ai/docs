# Installation Guidelines

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/mindformers/docs/source_en/installation.md)

## Confirming Version Matching Relationship

The currently supported hardware are the Atlas 800T A2, Atlas 800I A2, and Atlas 900 A3 SuperPoD.

The current recommended Python version for the suite is 3.11.4.

| MindSpore Transformers |                 MindSpore                 |                                                      CANN                                                      |                                               Firmware & Drivers                                                |
|:----------------------:|:-----------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.7.0          | [2.7.1](https://www.mindspore.cn/install) | [8.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) | [25.3.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0000.html) |

**The current MindSpore Transformers recommends the aforementioned software compatibility configuration. Concurrently, MindSpore Transformers remains compatible with the previous version of the MindSpore framework, enabling core functionalities to operate normally. However, features dependent upon the new framework version may not be accessible.**

Historical version matching relationship:

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                               Firmware & Drivers                                                |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.6.0          |   [2.7.0](https://www.mindspore.cn/install)   | [8.2.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html) |  [25.2.0](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0000.html)  |
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

## Installing Dependent Software

1. Install Firmware and Driver: Download the firmware and driver package through the [Confirming Version Matching Relationship](https://www.mindspore.cn/mindformers/docs/en/r1.7.0/installation.html#confirming-version-matching-relationship) to download the installation package, and refer to the [Ascend official tutorial](https://www.hiascend.com/en/document) for installation.

2. Install CANN and MindSpore: Follow the [Manual Installation](https://www.mindspore.cn/install/en) section on the MindSpore website for installation.

## Installing MindSpore Transformers

MindSpore Transformers supports both source code compiled installation and pip installation.

### Installation by Source Code Compilation

Users can compile and install MindSpore Transformers by executing the following command:

```bash
git clone -b r1.7.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

### Installation by pip

Users can install MindSpore Transformers by executing the following command:

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple
```

> - This installation method requires access to the public network. If you are in an internal network environment, please ensure that the network connection is configured correctly.
> - This method only installs the basic software package of MindSpore Transformers. Model files and scripts, etc., please obtain from the MindSpore Transformers Gitee repository.

## Installation Verification

To determine whether MindSpore Transformers has been successfully installed, execute the following code:

```bash
python -c "import mindformers as mf;mf.run_check()"
```

A similar result to the following proves that the installation was successful:

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```