# pip方式安装MindSpore CPU版本-macOS

<!-- TOC -->

- [pip方式安装MindSpore CPU版本-macOS](#pip方式安装mindspore-cpu版本-macos)
    - [安装Python](#安装python)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/install/mindspore_cpu_mac_install_pip.md)

## 安装Python

请参照[Python官网](https://www.python.org/)自行安装Python，版本要求为3.9-3.12。

安装完成后，可以通过以下命令查看Python版本。

```bash
python --version
```

## 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以2.7.2版本为例，执行以下命令。

```bash
export MS_VERSION=2.7.2
```

然后执行以下命令安装MindSpore。

```bash
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.7.2/setup.py)中的required_package），其余情况需自行安装依赖。

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
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
