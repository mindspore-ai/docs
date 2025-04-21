# Installing MindSpore CPU Nightly by pip-Windows

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip-Windows](#installing-mindspore-cpu-nightly-by-pip-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_cpu_win_install_nightly_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest changes on MindSpore.

This document describes how to install MindSpore Nightly by pip on Linux in a CPU environment.

## System Environment Information Confirmation

- Ensure that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Ensure that you have Python(>=3.9.0) installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/windows/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.

## Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/br_base/setup.py)). In other cases, you need to install dependency by yourself.
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

## Installation Verification

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
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
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
