# pip方式安装MindSpore CPU版本（Windows）

<!-- TOC -->

- [pip方式安装MindSpore CPU版本（Windows）](#pip方式安装mindspore-cpu版本windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [验证是否安装成功](#验证是否安装成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_win_install_pip.md)

本文档介绍如何在CPU环境的Windows系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Windows 10是x86架构64位操作系统。
- 确认安装Python 3.7.5版本。  
    - 如果未安装或者已安装其他版本的Python，则需从华为云下载[Python 3.7.5版本 64位](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe)进行安装。
- 安装Python完毕后，将Python和pip添加到系统环境变量。
    - 添加Python：控制面板->系统->高级系统设置->环境变量。双击系统变量中的Path，将`python.exe`的路径添加进去。
    - 添加pip：`python.exe`同一级目录中的`Scripts`文件夹即为Python自带的pip文件，将其路径添加到系统环境变量中即可。

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/windows_x64/mindspore-{version}-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)），其余情况需自行安装。  
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
