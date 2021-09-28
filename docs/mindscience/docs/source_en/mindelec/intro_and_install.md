# MindElec Introduction and Installation

<!-- TOC -->

- [MindElec Installation](#mindelec-installation)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installation](#installation)
        - [Installation by pip](#installation-by-pip)
        - [Installation by Source Code](#installation-by-source-code)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindscience/docs/source_en/mindelec/intro_and_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## MindElec Overview

Electromagnetic simulation refers to simulating the propagation characteristics of electromagnetic waves in objects or space through computation. It is widely used in scenarios such as mobile phone tolerance simulation, antenna optimization, and chip design. Conventional numerical methods, such as finite difference and finite element, require mesh segmentation and iterative computation. The simulation process is complex and the computation time is long, which cannot meet the product design requirements. With the universal approximation theorem and efficient inference capability, the AI method can improve the simulation efficiency.

MindElec is an AI electromagnetic simulation toolkit developed based on MindSpore. It consists of the electromagnetic model library, data build and conversion, simulation computation, and result visualization. End-to-end AI electromagnetic simulation is supported. Currently, Huawei has achieved phase achievements in the tolerance scenario of Huawei mobile phones. Compared with the commercial simulation software, the S parameter error of AI electromagnetic simulation is about 2%, and the end-to-end simulation speed is improved by more than 10 times.

This tutorial mainly introduces how to use MindElec, which is built in MindSpore, to perform high performance electromagnetic simulation using AI method.

> Here you can download the complete sample code: <https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples>.

In the future, MindElec will implement more simulation cases, and your contribution is welcome.

## MindElec Installation

### System Environment Information Confirmation

- The hardware platform should be Ascend, GPU or CPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.  
    The versions of MindElec and MindSpore must be consistent.
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/MindElec/requirements.txt).

### Installation

You can install MindElec either by pip or by source code.

#### Installation by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/mindscience/{arch}/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindscience/blob/master/MindElec/setup.py)），point cloud data sampling depends on [pythonocc](https://github.com/tpaviot/pythonocc-core), which you need to install manually.
> - `{arch}` specifies system architecture，for example, when using x86-64 Linux，`{arch}` should be x86_64, and aarch64 for ARM system(64-bit).
> - `{version}` specifies version of MindElec, 0.1.0 for example.
> - `{python_version}` specifies version of python, cp37-cp37m for python of version 3.7.5, and cp39-cp39 for python of version 3.9.0.

#### Installation by Source Code

1. Download source code from Gitee.

    ```bash
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. Run following command in source code directory, compile and install MindElec.

    ```bash
    cd ~/MindElec
    bash build.sh
    pip install output/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindelec'` when execute the following command:

```bash
python -c 'import mindelec'
```
