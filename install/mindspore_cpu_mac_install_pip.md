# pip方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [pip方式安装MindSpore CPU版本-macOS](#pip方式安装mindspore-cpu版本-macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_mac_install_pip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在macOS系统上使用pip方式快速安装MindSpore。

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

## 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以1.7.0版本为例，执行以下命令。

```bash
export MS_VERSION=1.7.0
```

然后根据系统架构及Python版本执行如下命令安装MindSpore。

```bash
# x86_64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp37-cp37m-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp38-cp38-linux_macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_10_15_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp38-cp38-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-macosx_11_0_arm64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

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
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
