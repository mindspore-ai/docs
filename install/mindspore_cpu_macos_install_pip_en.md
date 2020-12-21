# Installing MindSpore in CPU by pip (macOS)

<!-- TOC -->

- [Installing MindSpore in CPU by pip (macOS)](#installing-mindspore-in-cpu-by-pip-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_macos_install_pip_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

This document describes how to quickly install MindSpore by pip in a macOS system with a CPU environment.

## System Environment Information Confirmation

- Confirm that macOS Cata is installed with the 64-bit operating system.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) is installed.  
    - After installing, add the path of `python` to the environment variable PATH.

## Installing MindSpore

```bash
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)). In other cases, you need to manually install dependency items.  
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.

## Installation Verification

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

If the MindSpore version number is displayed, it means that MindSpore is installed successfully, and if the output is `No module named 'mindspore'`, it means that the installation was not successful.

## Version Update

Using the following command if you need to update MindSpore version:

```bash
pip install --upgrade mindspore
```
