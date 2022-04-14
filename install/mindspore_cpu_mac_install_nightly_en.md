# Installing MindSpore CPU Nightly by pip (macOS)

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip (macOS)](#installing-mindspore-cpu-nightly-by-pip-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_mac_install_pip_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.7/resource/_static/logo_source_en.png"></a>

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest features or bug fixes can use this version.

This document describes how to quickly install MindSpore Nightly on macOS by pip.

## System Environment Information Confirmation

- According to the system and chip situation in the table below to determine the appropriate Python version, macOS version and chip information can be found by clicking on the Apple logo in the upper left corner of the desktop - > `About this mac`:

    |Chip|Architecture|macOS Version|Supported Python Version|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.8-3.9|
    |Intel|x86_64|10.15/11.3|Python 3.7-3.9|

- Ensure that right Python version is installed. If not installed, download and install Python from:

    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg).
    - Python 3.8.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.8.0/python-3.8.0-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.8.0/python-3.8.0-macosx10.9.pkg)。
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg).

## Downloading and Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependencies are automatically downloaded during .whl package installation. (For details about the dependencies, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt).
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (rc) version, `{version}` should be specified manually, e.g. 1.5.0rc1; When updating to a standard release, `=={version}` could be removed.
