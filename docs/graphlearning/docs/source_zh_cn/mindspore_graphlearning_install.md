# 安装 Graph Learning

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

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/graphlearning/docs/source_zh_cn/mindspore_graphlearning_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 安装指南

### 确认系统环境信息

- 硬件平台确认为Linux系统下的GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.6.0版本。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/r0.2.0/requirements.txt)。

### 安装方式

可以采用pip安装或者源码编译安装两种方式。

#### pip安装

- Ascend/CPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0a0/GraphLearning/cpu/{system_structure}/mindspore_gl-0.2.0a0-cp37-cp37m-linux_{system_structure}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- GPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0a0/GraphLearning/gpu/x86_64/cuda-{cuda_verison}/mindspore_gl-0.2.0a0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

> - 在联网状态下，安装whl包时会自动下载MindSpore Graph Learning安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/graphlearning/blob/r0.2.0/requirements.txt)），其余情况需自行安装。
> - `{system_structure}`表示为Linux系统架构，可选项为`x86_64`和`arrch64`。
> - `{cuda_verison}`表示为CUDA版本，可选项为`10.1`、`11.1`和`11.6`。

#### 源码安装

1. 从代码仓下载源码

    ```bash
    git clone https://gitee.com/mindspore/graphlearning.git -b r0.2.0
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
