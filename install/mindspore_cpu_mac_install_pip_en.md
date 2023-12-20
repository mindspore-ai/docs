# Installing MindSpore in CPU by pip-macOS

<!-- TOC -->

- [Installing MindSpore in CPU by pip-macOS](#installing-mindspore-in-cpu-by-pip-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/install/mindspore_cpu_mac_install_pip_en.md)

This document describes how to install MindSpore on macOS by pip.

## System Environment Information Confirmation

- According to the system and chip situation in the table below to determine the appropriate Python version, macOS version and chip information can be found by clicking on the Apple logo in the upper left corner of the desktop - > `About this mac`:

    |Chip|Architecture|macOS Version|Supported Python Version|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.8-3.9|
    |Intel|x86_64|10.15/11.3|Python 3.7-3.9|

> Note: [Python 3.8.10](https://www.python.org/downloads/release/python-3810/) or Python 3.8.5 via Conda are the oldest Python releases that supports macOS for ARM

- Ensure that a correct version of Python is installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/macos/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.

## Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.2.10 as an example, execute the following commands.

```bash
export MS_VERSION=2.2.10
```

Then run the following commands to install MindSpore according to the system architecture and Python version.

```bash
# x86_64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp37-cp37m-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp38-cp38-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp38-cp38-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. For details about dependencies, see required_package in the [setup.py](https://gitee.com/mindspore/mindspore/blob/r2.2/setup.py). In other cases, install the dependencies by yourself. When running a model, you need to install additional dependencies based on the requirements.txt file specified by different models in the [ModelZoo](https://gitee.com/mindspore/models/tree/r2.2/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.2/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
