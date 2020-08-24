# MindSpore API概述

<!-- TOC -->

- [MindSpore API概述](#mindsporeapi概述)
    - [设计理念](#设计理念)
    - [层次结构](#层次结构)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/api_structure.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 设计理念

MindSpore源于全产业的最佳实践，向数据科学家和算法工程师提供了统一的模型训练、推理和导出等接口，支持端、边、云等不同场景下的灵活部署，推动深度学习和科学计算等领域繁荣发展。

MindSpore提供了动态图和静态图统一的编码方式，用户无需开发多套代码，仅变更一行代码便可切换动态图/静态图模式，从而拥有更轻松的开发调试及性能体验。

此外，由于MindSpore统一了单机和分布式训练的编码方式，开发者无需编写复杂的分布式策略，在单机代码中添加少量代码即可实现分布式训练，大大降低了AI开发门槛。

## 层次结构

MindSpore向用户提供了3个不同层次的API，支撑用户进行网络构建、整图执行、子图执行以及单算子执行，从低到高分别为Low-Level Python API、Medium-Level Python API以及High-Level Python API。

![img](./images/api_structure.png) 

- Low-Level Python API

  第一层为低阶API，主要包括张量定义、基础算子、自动微分等模块，用户可使用低阶API轻松实现张量操作和求导计算。

- Medium-Level Python API

  第二层为中阶API，其封装了低价API，提供网络层、优化器、损失函数等模块，用户可通过中阶API灵活构建神经网络和控制执行流程，快速实现模型算法逻辑。

- High-Level Python API

  第三层为高阶API，其在中阶API的基础上又提供了训练推理的管理、Callback、混合精度训练等高级接口，方便用户控制整网的执行流程和实现神经网络的训练及推理。
