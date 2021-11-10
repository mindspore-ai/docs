# pip方式安装MindSpore CPU版本（Windows）

<!-- TOC -->

- [pip方式安装MindSpore CPU版本（Windows）](#pip方式安装mindspore-cpu版本windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_cpu_win_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用pip方式快速安装MindSpore。

在确认系统环境信息的过程中，如需了解如何安装第三方依赖软件，可以参考社区提供的实践——[在Windows（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=364)中的第三方依赖软件安装相关部分，在此感谢社区成员[lvmingfu](https://gitee.com/lvmingfu)的分享。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/python-3.9.0-amd64.exe)。
- 安装Python完毕后，将Python和pip添加到系统环境变量。
    - 添加Python：控制面板->系统->高级系统设置->环境变量。双击系统变量中的Path，将`python.exe`的路径添加进去。
    - 添加pip：`python.exe`同一级目录中的`Scripts`文件夹即为Python自带的pip文件，将其路径添加到系统环境变量中即可。

## 安装MindSpore

参考[版本列表](https://www.mindspore.cn/versions)先进行SHA-256完整性校验，校验一致后再执行如下命令安装MindSpore。

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/x86_64/mindspore-{version}-{python_version}-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.5/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.5/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.5/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.5.0版本MindSpore时，`{version}`应写为1.5.0，而安装1.5.0-rc1版本时，第一个`{version}`代表下载路径应写为1.5.0-rc1，第二个`{version}`代表版本号应写为1.5.0rc1。
- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

## 验证是否成功安装

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
mindspore version: 版本号
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

