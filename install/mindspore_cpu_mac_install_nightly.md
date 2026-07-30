# pip方式安装MindSpore CPU Nightly版本-macOS

<!-- TOC -->

- [pip方式安装MindSpore CPU Nightly版本-macOS](#pip方式安装mindspore-cpu-nightly版本-macos)
    - [安装Python](安装python)
    - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_pip.md)

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

## 安装Python

请参照[Python官网](https://www.python.org/)自行安装Python，版本要求为3.10-3.12。

安装完成后，可以通过以下命令查看Python版本。

```bash
python --version
```

## 下载安装MindSpore

执行以下命令安装MindSpore：

```bash
# install prerequisites
conda install scipy -c conda-forge

pip install mindspore-dev -i https://repo.huaweicloud.com/repository/pypi/simple/
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://atomgit.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装依赖。
- pip会自动安装当前最新版本的MindSpore Nightly，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

## 验证是否成功安装

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行以下命令：

```bash
pip install --upgrade mindspore-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
