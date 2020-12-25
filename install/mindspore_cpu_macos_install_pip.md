# pip方式安装MindSpore CPU版本（macOS）

<!-- TOC -->

- [pip方式安装MindSpore CPU版本（macOS）](#pip方式安装mindspore-cpu版本macOS)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_macos_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的macOS系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装macOS Catalina是64位操作系统。
- 确认安装[Python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg)版本。  
- 安装Python完毕后，将Python添加到系统环境变量。
    - 将Python路径添加到系统环境变量中即可。

## 安装MindSpore

```bash
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。

## 验证是否安装成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未安装成功。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore
```
