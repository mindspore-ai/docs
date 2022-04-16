# pip方式安装MindSpore CPU Nightly版本-macOS

<!-- TOC -->

- [pip方式安装MindSpore CPU Nightly版本-macOS](#pip方式安装mindspore-cpu-nightly版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_pip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在macOS系统上使用pip方式快速安装MindSpore Nightly。

## 确认系统环境信息

- 根据下表中的系统及芯片情况确定合适的Python版本，macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.8-3.9|
    |Intel|x86_64|10.15/11.3|Python 3.7-3.9|

- 确认安装对应的Python版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5 (64-bit)：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg)。
    - Python 3.8.0 (64-bit)：[官网](https://www.python.org/ftp/python/3.8.0/python-3.8.0-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.8.0/python-3.8.0-macosx10.9.pkg)。
    - Python 3.9.0 (64-bit)：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg)。

## 下载安装MindSpore

执行如下命令安装MindSpore：

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- pip会自动安装当前最新版本的Nightly版本MindSpore，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

## 验证是否成功安装

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
