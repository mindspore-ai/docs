# Installing MindSpore in CPU by pip-macOS

<!-- TOC -->

- [Installing MindSpore in CPU by pip-macOS](#installing-mindspore-in-cpu-by-pip-macos)
    - [Installing Python](#installing-python)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/install/mindspore_cpu_mac_install_pip_en.md)

## Installing Python

Please refer to the [Python official website](https://www.python.org/) to install Python by yourself. The version must be 3.9-3.11.

After the installation is complete, you can check the Python version with the following command.

```bash
python --version
```

## Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.7.1 as an example, execute the following commands.

```bash
export MS_VERSION=2.7.1
```

Then run the following command to install MindSpore.

```bash
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. For details about dependencies, see required_package in the [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.7.1/setup.py). In other cases, install the dependencies by yourself.

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
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
