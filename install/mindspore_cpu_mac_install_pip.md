# pip方式安装MindSpore CPU版本（macOS）

<!-- TOC -->

- [pip方式安装MindSpore CPU版本（macOS）](#pip方式安装mindspore-cpu版本macos)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_mac_install_pip.md)

本文档介绍如何在macOS系统上使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 根据下表中的系统及芯片情况确定合适的Python版本，macOS版本及芯片信息可点击桌面左上角苹果标志->`关于本机`获悉：

    |芯片|计算架构|macOS版本|支持Python版本|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.9.1+（M1当前不支持3.7, 3.9支持最低版本为3.9.1）|
    |Intel|x86_64|10.15/11.3|Python 3.7.5/Python 3.9.0|

- 确认安装对应的Python版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5 (64-bit)：[官网](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg)。
    - Python 3.9.0 (64-bit)：[官网](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) 或者 [华为云](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg)。
    - Python 3.9.1 (64-bit)：[官网](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg) 或者 [华为云](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg)。

## 安装MindSpore

参考[版本列表](https://www.mindspore.cn/versions)先进行SHA-256完整性校验，校验一致后再执行如下命令安装MindSpore。

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/{arch}/mindspore-{version}-{python_version}-macosx_{platform_version}_{platform_arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.6/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.5.0版本MindSpore时，`{version}`应写为1.5.0，而安装1.5.0-rc1版本时，第一个`{version}`代表下载路径应写为1.5.0-rc1，第二个`{version}`代表版本号应写为1.5.0rc1。
- `{arch}`表示架构，例如使用的macOS系统是x86架构64位时，`{arch}`应写为`x86_64`。如果是M1芯片，则写为`aarch64`。
- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。
- `{platform_version}`表示系统版本，如果是M1芯片，`{platform_version}`应写为`11_0`，否则`{platform_version}`应写为`10_15`。
- `{platform_arch}`表示系统架构，例如使用的macOS系统是x86架构64位时，`{platform_arch}`应写为`x86_64`。如果是M1芯片，则写为`arm64`。

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
