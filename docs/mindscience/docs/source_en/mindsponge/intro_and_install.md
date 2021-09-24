# Introduction and Installation of MindSPONGE

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindscience/docs/source_en/mindsponge/intro_and_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## MindSPONGE Overview

Molecular simulation is a method of exploiting computer to simulate the structure and behavior of molecules by using the molecular model at the atomic-level, and then simulate the physical and chemical properties of the molecular system. It builds a set of models and algorithms based on the experiment and through the basic principles, so as to calculate the reasonable molecular structure and molecular behavior.

In recent years, molecular simulation technology has been developed rapidly and widely used in many fields. In the field of medical design, it can be used to study the mechanism of action of virus and drugs. In the field of biological science, it can be used to characterize the multi-level structure and properties of proteins. In the field of materials science, it can be used to study the structure and mechanical properties, material optimization design. In the field of chemistry, it can be used to study surface catalysis and mechanism. In the field of petrochemical industry, it can be used for structure characterization, synthesis design, adsorption and diffusion of molecular sieve catalyst, construction and characterization of polymer chain and structure of crystalline or amorphous bulk polymer, and prediction of important properties including blending behavior, mechanical properties, diffusion, cohesion and so on.

MindSPONGE is molecular simulation library jointly developed by the `Gao Yiqin` research group of PKU and Shenzhen Bay Laboratory and Huawei `MindSpore` team, which is the combination of `MindSpore` and `SPONGE`(`S`imulation `P`ackage `O`f `N`ext `GE`neration molecular modeling). MindSPONGE has the features like high-performance, modularization, etc. MindSPONGE can complete the traditional molecular simulation process efficiently based on MindSpore's automatic parallelism, graph-computing fusion and other features. MindSPONGE can combine AI methods such as neural networks with traditional molecular simulations by utilizing MindSpore's feature of automatic differentiation.

This tutorial mainly introduces how to use SPONGE, which is built in MindSpore, to perform high performance molecular simulation on the GPU.

> Here you can download the complete sample code: <https://gitee.com/mindspore/mindscience/tree/r0.1/MindSPONGE/examples>.

In the future, MindSPONGE will implement more simulation cases, and your contribution is welcome.

## Installation

### Confirming System Environment Information

- The hardware platform should be Linux CPU with avx supported.
- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install), install MindSpore, version 1.5.0 is required.

### Install by pip

#### Install MindSpore

```bash
pip install mindspore-gpu
```

#### Install MindSPONGE

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/mindscience/x86_64/mindscience_mindsponge_gpu-{version}-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindscience/blob/r0.1/MindSPONGE/setup.py)). In other cases, you need to manually install dependency items.
> - `{version}` refers to the MindSPONGE version. For example, if you want to install MindSPONGE 0.1.0, {version} should be 0.1.0.

### Install by Source Code

1. Download Source Code from Gitee

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git -b r0.1
    ```

2. Compiling MindSPONGE

    ```bash
    cd ~/MindScience/MindSPONGE
    python setup.py install --user
    ```

### Verifying Successful Installation

Successfully installed, if there is no error message such as No module named 'mindsponge' when execute the following command:

```bash
python -c 'import mindsponge'
```
