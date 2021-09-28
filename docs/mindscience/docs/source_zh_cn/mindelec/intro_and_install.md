# MindElec介绍和安装

<!-- TOC -->

- [安装MindELec](#安装mindelec)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->
<a href=
"https://gitee.com/mindspore/docs/blob/r1.5/docs/mindscience/docs/source_zh_cn/mindelec/intro_and_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## MindElec介绍

电磁仿真是指通过计算的方式模拟电磁波在物体或空间中的传播特性，其在手机容差、天线优化和芯片设计等场景中应用广泛。传统数值方法如有限差分、有限元等需网格剖分、迭代计算，仿真流程复杂、计算时间长，无法满足产品的设计需求。AI方法具有万能逼近能力和高效推理能力，可有效提升仿真效率。

MindElec是基于MindSpore开发的AI电磁仿真工具包，由数据构建及转换、仿真计算、以及结果可视化组成。可以支持端到端的AI电磁仿真。目前已在华为终端手机容差场景中取得阶段性成果，相比商业仿真软件，AI电磁仿真的S参数误差在2%左右，端到端仿真速度提升10+倍。

MindElec中包含了多个AI电磁仿真案例，更多详情，请点击查看[案例](https://gitee.com/mindspore/mindscience/tree/r0.1/MindElec/examples)。

未来，MindElec中将包含更多结合AI算法的电磁仿真案例，欢迎大家的关注和支持。

## MindElec安装

### 确认系统环境信息

- 硬件平台为Ascend。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。  
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/blob/r0.1/MindElec/requirements.txt)。

### 安装方式

可以采用pip安装或者源码编译安装两种方式。

#### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/mindscience/{arch}/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindElec安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindscience/blob/r0.1/MindElec/setup.py)），点云数据采样依赖[pythonocc](https://github.com/tpaviot/pythonocc-core)，需自行安装。
> - `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为x86_64。如果系统是ARM架构64位，则写为aarch64。
> - `{version}`表示MindElec版本号，例如下载0.1.0版本MindElec时，`{version}`应写为0.1.0。
> - `{python_version}`表示用户的Python版本，Python版本为3.7.5时，{python_version}应写为cp37-cp37m。Python版本为3.9.0时，则写为cp39-cp39。

#### 源码安装

1. 从Gitee下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. 在源码根目录下，执行如下命令编译并安装MindElec。

    ```bash
    cd ~/MindElec
    bash build.sh
    pip install output/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindelec'`，则说明安装成功。

```bash
python -c 'import mindelec'
```
