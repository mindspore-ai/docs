# Installing MindSpore in CPU by Source Code-macOS

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code-macOS](#installing-mindspore-in-cpu-by-source-code-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_source_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by compiling source code on macOS.

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

- Ensure that [Xcode](https://xcodereleases.com/) (>=12.4 and <= 13.0) is installed, where 12.4(X86) and 13.0(m1) are verified.

- Ensure that `Command Line Tools for Xcode` is installed. If not, use `sudo xcode-select --install` command to install Command Line Tools.

- Ensure that [CMake 3.18.3 and the later version](https://cmake.org/download/) is installed. Use `brew install cmake` if it's not installed.

- Ensure that [patch 2.5](https://ftp.gnu.org/gnu/patch/) is installed. Use `brew install patch` if it's not installed.

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed. Use `pip install wheel` if it's not installed.

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4   # -j is a thread configuration when compiled, and if cpu performance is better, compile using multithreading, with parameters usually twice the number of CPU cores
```

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If there is any compile error when installing scipy package, please use the following command to install scipy package first, then install mindspore package.

```bash
pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy
```

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

Using the following command if you need to update the MindSpore version:

- Update online directly

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compilation script `build.sh` in the source code root directory, find the generated whl installation package by compilation in the `output` directory, and then execute the command to upgrade.

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-macosx_{platform_version}_{arch}.whl
    ```
