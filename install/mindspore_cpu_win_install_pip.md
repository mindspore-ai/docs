# pip方式安装MindSpore CPU版本-Windows

<!-- TOC -->

- [pip方式安装MindSpore CPU版本-Windows](#pip方式安装mindspore-cpu版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/install/mindspore_cpu_win_install_pip.md)

本文档介绍如何在CPU环境的Windows系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装Python（>=3.7.5）。可以从[Python官网](https://www.python.org/downloads/windows/)或者[华为云](https://repo.huaweicloud.com/python/)选择合适的版本进行安装。

## 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以2.2.12版本为例，执行以下命令。

```bash
set MS_VERSION=2.2.12
```

然后根据Python版本执行如下命令安装MindSpore。

```bash
# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp39-cp39-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r2.2/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r2.2/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.2/requirements.txt)。

## 验证是否成功安装

```bash
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
