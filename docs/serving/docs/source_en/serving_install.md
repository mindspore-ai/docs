# MindSpore Serving Installation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_en/serving_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

<!-- TOC -->

- [MindSpore Serving Installation](#mindspore-serving-installation)
    - [Installation](#installation)
        - [Installation by pip](#installation-by-pip)
        - [Installation by Source Code](#installation-by-source-code)
    - [Installation Verification](#installation-verification)
    - [Configuring Environment Variables](#configuring-environment-variables)

<!-- /TOC -->

## Installation

MindSpore Serving supports the Ascend 310, Ascend 910 and Nvidia GPU environments.

MindSpore Serving depends on the MindSpore training and inference framework. Therefore, install [MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) and then MindSpore Serving. You can install MindInsight either by pip or by source code.

### Installation by pip

Perform the following steps to install Serving:

If use the pip command, download the .whl package from the [MindSpore Serving page](https://www.mindspore.cn/versions/en) and install it.

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Serving/{arch}/mindspore_serving-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}` denotes the version of MindSpore Serving. For example, when you are downloading MindSpore Serving 1.1.0, `{version}` should be 1.1.0.
> - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.
> - `{python_version}` spcecifies the python version for which MindSpore is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`. Please use the same Python environment whereby MindSpore is installed.

### Installation by Source Code

Download the [source code](https://gitee.com/mindspore/serving) and go to the `serving` directory.

Method 1: Specify the path of the installed or built MindSpore package on which Serving depends and install Serving.

```shell
sh build.sh -p $MINDSPORE_LIB_PATH
```

In the preceding information, `build.sh` is the build script file in the `serving` directory, and `$MINDSPORE_LIB_PATH` is the `lib` directory in the installation path of the MindSpore software package, for example, `softwarepath/mindspore/lib`. This path contains the library files on which MindSpore depends.

Method 2: Directly build Serving. The MindSpore package is built together with Serving. You need to configure the [environment variables](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables) for MindSpore building.

```shell
# GPU
sh build.sh -e gpu
# Ascend 310 and Ascend 910
sh build.sh -e ascend
```

In the preceding information, `build.sh` is the build script file in the `serving` directory. After the build is complete, find the .whl installation package of MindSpore in the `serving/third_party/mindspore/build/package/` directory and install it.

```shell
pip install mindspore_ascend-{version}-{python_version}-linux_{arch}.whl
```

Find the .whl installation package of Serving in the `serving/build/package/` directory and install it.

```shell
pip install mindspore_serving-{version}-{python_version}-linux_{arch}.whl
```

## Installation Verification

Run the following commands to verify the installation. Import the Python module. If no error is reported, the installation is successful.

```python
from mindspore_serving import server
```

### Configuring Environment Variables

To run MindSpore Serving, configure the following environment variables:

- MindSpore Serving depends on MindSpore. You need to configure [environment variables](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables) to run MindSpore.
