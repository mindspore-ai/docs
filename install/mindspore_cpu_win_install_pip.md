# Pip方式安装MindSpore

<!-- TOC -->

- [Pip方式安装MindSpore](#pip方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_win_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Windows系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 需要确认您的Windows 7/8/10是64位操作系统。
- 确认安装Python 3.7.5版本。  
    如果未安装或者安装其他版本的Python，则需从华为云下载[python 3.7.5版本 64位](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe)进行安装。
- 安装Python完毕后，将Python和Pip添加到系统环境变量。
    - Python添加：控制面板->系统->高级系统设置->环境变量。双击系统变量中的Path，将`python.exe`的路径添加进去。
    - Pip添加：`python.exe`同一级目录中的`Scripts`文件夹即为Python自带的pip文件，将其路径添加到系统环境变量中即可。

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/windows_x64/mindspore-{version}-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。

## 验证是否安装成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明安装未成功。
