# MindSPONGE

- [MindSPONGE介绍](#MindSPONGE介绍)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装](#安装)
        - [安装MindSpore](#安装mindspore)
        - [安装MindSPONGE](#安装MindSPONGE)
    - [源码安装](#源码安装)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [许可证](#许可证)

## MindSPONGE介绍

分子模拟是指利用计算机以原子水平的分子模型来模拟分子结构与行为，进而模拟分子体系的各种物理、化学性质的方法。它是在实验基础上，通过基本原理，构筑起一套模型和算法，从而计算出合理的分子结构与分子行为。

近年来，分子模拟技术发展迅速并且在多个学科领域得到了广泛的应用。在药物设计领域，可用于研究病毒、药物的作用机理等；在生物科学领域，可用于表征蛋白质的多级结构与性质；在材料学领域，可用于研究结构与力学性能、材料的优化设计等；在化学领域，可用于研究表面催化及机理；在石油化工领域，可用于分子筛催化剂结构表征、合成设计、吸附扩散，可构建和表征高分子链以及晶态或非晶态本体聚合物的结构，预测包括共混行为、机械性质、扩散、内聚等重要性质。

MindSPONGE是由`高毅勤`课题组（北京大学、深圳湾实验室）和华为`MindSpore`团队联合开发的分子模拟库，具有高性能、模块化等特性。MindSPONGE是`MindSpore`和`SPONGE`（`S`imulation `P`ackage `O`f `N`ext `GE`neration molecular modeling）的缩写。MindSPONGE是第一个根植于AI计算框架的分子模拟工具，其使用模块化的设计思路，可以快速构建分子模拟流程，并且基于MindSpore自动并行、图算融合等特性，可高效地完成传统分子模拟。同时，MindSPONGE也可以将神经网络等AI方法与传统分子模拟进行结合，应用到生物、材料、医药等领域中。

MindSPONGE中包含了多个传统分子模拟案例，更多详情，请点击查看[案例](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/examples)。

未来，MindSPONGE中将包含更多结合AI算法的分子模拟案例，欢迎大家的关注和支持。

## 安装教程

### 确认系统环境信息

- 硬件平台确认为Linux系统下的GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.2.0版本。

### pip安装

#### 安装MindSpore

```bash
pip install mindspore-gpu
```

#### 安装MindSPONGE

```bash
cd ~/MindScience/MindSPONGE
bash build.sh
pip install ./output/mindsponge-*.whl
```

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. 编译安装MindSPONGE

    ```bash
    cd ~/MindScience/MindSPONGE
    python setup.py install --user
    ```

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

## 贡献

欢迎参与贡献MindSPONGE。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 许可证

[Apache License 2.0](LICENSE)
