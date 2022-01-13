# Installing MindSpore in CPU by Source Code (macOS)

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code (macOS)](#installing-mindspore-in-cpu-by-source-code-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by compiling source code on macOS.

## System Environment Information Confirmation

- Ensure macOS version between 10.15 and 11.3, for M1 chip currently only support 11.3.

- Ensure [Xcode](https://xcodereleases.com/) (>=12.4 and <= 13.0) is installed, where 12.4(X86) and 13.0(m1) are verified.

- Ensure Command Line Tools for Xcode is installed. If not, use `sudo xcode-select --install` command to install it.

- Ensure that Python 3.7 or 3.9 is installed. If not installed, download and install Python from:

    - Python 3.7.5 (64-bit for macOS 10.15)：[Python official website](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg).
    - Python 3.9.0 (64-bit for macOS 10.15)：[Python official website](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg).
    - Python 3.9.1 (64-bit for macOS 11.3)：[Python official website](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg) or [HUAWEI CLOUD](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg).

- Ensure that [CMake](https://cmake.org/download/) > `3.18.3` is installed. Use `brew install cmake` if it's not installed.

- Ensure that [patch 2.5](http://ftp.gnu.org/gnu/patch/) is installed. Use `brew install patch` if it's not installed.

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
bash build.sh -e cpu -S on -j4   # -j stands for the thread number can be compiling with, can assign twice as much as CPU cores
```

The artifact of MindSpore should lie in directory `output/` within the repo directory.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

>If there is any compile error when installing scipy package, please use `pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy` to install scipy package first, then install mindspore package as normal.

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

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-macosx_{platform_version}_{arch}.whl
    ```
