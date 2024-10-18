# Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindformers/docs/source_en/quick_start/install.md)

## Version Matching Relationship

The currently supported hardware is the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server.

The current recommended Python version for the suite is 3.10.

| MindFormers |                 MindSpore                  | CANN | Firmware & Drivers | Mirror Links |
|:-----------:|:------------------------------------------:|:----:|:------------------:|:------------:|
|   r1.3.0    | [2.4.0](https://www.mindspore.cn/install/) | TBD  |        TBD         |     TBD      |

**Currently MindFormers recommends using a software package relationship as above.**

### Compatibility Notes

MindFormers has the following compatible relationships with MindSpore:

| MindFormers | MindSpore | Compatibility |
|:-----------:|:---------:|:-------------:|
|    1.3.0    |    2.3    |       √       |
|    1.2.0    |    2.4    |       √       |

## Environment Installation

1. Install Firmware and Driver: Download the firmware and driver package through the [Version Matching Relationship](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/quick_start/install.html#version-matching-relationship) to download the installation package, and refer to the [Ascend official tutorial](https://www.hiascend.com/document/detail/zh/quick-installation/24.0.RC1/quickinstg_train/800_9000A2/quickinstg_800_9000A2_0007.html) for installation.

2. Install CANN and MindSpore: Use the officially provided Docker image (CANN, MindSpore are already included in the image, no need to install them manually) or follow the [Manual Installation](https://www.mindspore.cn/install/en#manual-installation) section on the MindSpore website for installation.

## MindFormers Installation

Currently only source code compilation installation is supported, users can execute the following command to install MindFormers:

```bash
git clone -b r1.3.0 https://gitee.com/mindspore/mindformers.git
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