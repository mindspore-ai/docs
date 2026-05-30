# LiteBoost 简介

## 概述

LiteBoost是MindSpore Lite面向昇腾硬件的推理加速工具包，提供高性能自定义算子、多卡并行推理、量化稀疏等推理加速能力。LiteBoost基于PyTorch接口，通过C++自定义算子深度调用昇腾CANN `aclnn`接口。同时结合Python层的优化Attention、RoPE实现以及HCCL多卡通信，实现端到端推理加速。

## 核心能力

### 高性能自定义算子

- 通过对接CANN包融合算子，提供易用接口，快速使用融合算子，提升模型推理性能。
- 通过开发自定义融合算子，基于LiteBoost提供对外接口，基于PyTorch提升模型推理性能。

### 多卡并行

- 支持多卡TP、CP、SP、DP等多种并行策略。
- 通过不同的并行策略优化适配不同算法模型，针对开源模型提供简单易用的使用方式，提升开发者使能多卡并行能力。

## 技术架构

LiteBoost采用**C++算子层 + Python加速层**的双层架构：

- **C++算子层**：通过PyTorch `TORCH_LIBRARY`机制注册自定义算子，编译为共享库，深度调用昇腾CANN `aclnn`接口，充分发挥昇腾NPU硬件性能，会持续开发自定义算子，提升该组件的推理性能。
- **Python加速层**：封装C++算子的Python绑定、优化后的Attention等Layer、以及基于HCCL的多卡并行方案，提供简洁易用的Python API，后续会持续更新以及新增量化稀疏相关加速优化能力。
