# Installing MindSpore CPU Nightly by pip-macOS

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip-macOS](#installing-mindspore-cpu-nightly-by-pip-macos)
    - [Installing Python](#installing-python)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_pip_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest features or bug fixes can use this version.

## Installing Python

Please refer to the [Python official website](https://www.python.org/) to install Python by yourself. The version requirement is 3.10-3.12.

After the installation, you can check the Python version with the following command.

```bash
python --version
```

## Downloading and Installing MindSpore

Execute the following command to install MindSpore:

```bash
# install prerequisites
conda install scipy -c conda-forge

pip install mindspore-dev -i https://repo.huaweicloud.com/repository/pypi/simple/
```

Of which,

- When the network is connected, dependencies are automatically downloaded during .whl package installation. (For details about the dependencies, see required_package in [setup.py](https://atomgit.com/mindspore/mindspore/blob/master/setup.py)). In other cases, you need to install dependencies by yourself.
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
