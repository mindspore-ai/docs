# MindSpore Elec Introduction and Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindelec/docs/source_en/intro_and_install.md)&nbsp;&nbsp;

## MindSpore Elec Overview

Electromagnetic simulation refers to simulating the propagation characteristics of electromagnetic waves in objects or space through computation. It is widely used in scenarios such as mobile phone tolerance simulation, antenna optimization, and chip design. Conventional numerical methods, such as finite difference and finite element, require mesh segmentation and iterative computation. The simulation process is complex and the computation time is long, which cannot meet the product design requirements. With the universal approximation theorem and efficient inference capability, the AI method can improve the simulation efficiency.

MindSpore Elec is an AI electromagnetic simulation toolkit developed based on MindSpore. It consists of the electromagnetic model library, data build and conversion, simulation computation, and result visualization. End-to-end AI electromagnetic simulation is supported. Currently, Huawei has achieved phase achievements in the tolerance scenario of Huawei mobile phones. Compared with the commercial simulation software, the S parameter error of AI electromagnetic simulation is about 2%, and the end-to-end simulation speed is improved by more than 10 times.

MindSpore Elec contains several AI EM simulation cases. For more details, please click to view [case](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples).

In the future, MindSpore Elec will implement more simulation cases, and your contribution is welcome.

## MindSpore Elec Installation

### System Environment Information Confirmation

- The hardware platform should be Ascend, GPU or CPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.  
    The versions of MindSpore Elec and MindSpore must be consistent.
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/MindElec/requirements.txt).

### Installation

You can install MindSpore Elec either by pip or by source code.

#### Installation by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/mindscience/{arch}/mindscience_mindelec_ascend-{me_version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindscience/blob/master/MindElec/setup.py)), point cloud data sampling depends on [pythonocc](https://github.com/tpaviot/pythonocc-core), which you need to install manually.
> - `{arch}` specifies system architecture，for example, when using x86-64 Linux，`{arch}` should be x86_64, and aarch64 for ARM system(64-bit).
> - `{ms_version}` refers to the MindSpore version that matches with MindSpore Elec. For example, if you want to install MindSpore Elec 0.1.0, then,`{ms_version}` should be 1.5.0。
> - `{me_version}` refers to the version of MindSpore Elec. For example, when you are downloading MindSpore Elec 0.1.0, `{me_version}` should be 0.1.0.
> - `{python_version}` specifies version of python, cp37-cp37m for python of version 3.7.5, and cp39-cp39 for python of version 3.9.0.

#### Installation by Source Code

1. Download source code from Gitee.

    ```bash
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. Run following command in source code directory, compile and install MindSpore Elec.

    ```bash
    cd mindscience/MindElec
    bash build.sh
    pip install output/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindelec'` when execute the following command:

```bash
python -c 'import mindelec'
```
