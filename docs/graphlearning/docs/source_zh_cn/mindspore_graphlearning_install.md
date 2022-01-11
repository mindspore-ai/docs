# MindSpore Graph Learning

- [MindSpore Graph Learning介绍](#mindspore-graph-learning介绍)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/graphlearning/docs/source_zh_cn/mindspore_graphlearning_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## MindSpore Graph Learning介绍

MindSpore Graph Learning是一个基于MindSpore的高效易用的图学习框架。

![GraphLearning_architecture](./images/MindSpore_GraphLearning_architecture.PNG)

相较于一般模型，图神经网络模型需要在给定的图结构上做信息的传递和聚合，现有系统无法直观表达这些操作。MindSpore Graph Learning创新提出以点为中心的编程范式，更符合图学习算法逻辑和Python语言风格，减少算法设计和实现间的差距。

同时，结合MindSpore的图算融合和自动算子编译技术（AKG）特性，自动识别图神经网络任务特有执行pattern进行融合和kernel level优化，能够覆盖现有框架中已有的算子和新组合算子的融合优化，获得相比现有流行框架平均3到4倍的性能提升。

结合MindSpore深度学习框架，框架基本能够覆盖大部分的图神经网络应用，详情请参考<https://gitee.com/mindspore/graphlearning/tree/r0.5/model_zoo>。

## 安装指南

### 确认系统环境信息

- 硬件平台确认为Linux系统下的GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.6.0版本。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/r0.5/requirements.txt)。

### 安装方式

可以采用pip安装或者源码编译安装两种方式。

#### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/GraphLearning/any/mindspore_gl_gpu-{version}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Graph Learning安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/r0.5/requirements.txt)），其余情况需自行安装。
> - `{version}`表示MindSpore Graph Learning版本号，例如下载0.1版本MindSpore Graph Learning时，`{version}`应写为0.1。

#### 源码安装

1. 从代码仓下载源码

    ```bash
    git clone https://gitee.com/mindspore/graphlearning.git -b r0.1
    ```

2. 编译安装MindSpore Graph Learning

    ```bash
    cd graphlearning
    bash build.sh
    pip install ./output/mindspore_gl_gpu-*.whl
    ```

### 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindspore_gl'`，则说明安装成功。

```bash
python -c 'import mindspore_gl'
```
