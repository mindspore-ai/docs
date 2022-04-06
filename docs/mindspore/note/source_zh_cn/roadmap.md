# 路标

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/note/source_zh_cn/roadmap.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

以下将展示MindSpore近一年的高阶计划，我们会根据用户的反馈诉求，持续调整计划的优先级。

总体而言，我们会努力在以下几个方面不断改进。

1. 提供更多的预置模型支持。
2. 持续补齐API和算子库，改善易用性和编程体验。
3. 提供华为昇腾AI处理器的全面支持，并不断优化性能及软件架构。
4. 完善可视化、调试调优、安全相关工具。

热忱希望各位在用户社区加入讨论，并贡献您的建议。

## 预置模型

- CV：目标检测、GAN、图像分割、姿态识别等场景经典模型。
- NLP：RNN、Transformer类型神经网络，拓展基于Bert预训练模型的应用。
- 其它：GNN、强化学习、概率编程、AutoML等。

## 易用性

- 补齐算子、优化器、Loss函数等各类API
- 完善Python语言原生表达支持
- 支持常见的Tensor/Math操作
- 增加更多的自动并行适用场景，提高策略搜索的准确性

## 性能优化

- 优化编译时间
- 低比特混合精度训练/推理
- 提升内存使用效率
- 提供更多的融合优化手段
- 加速PyNative执行性能

## 架构演进

- 图算融合优化：使用细粒度Graph IR表达算子，构成带算子边界的中间表达，挖掘更多图层优化机会。
- 支持更多编程语言
- 优化数据增强的自动调度及分布式训练数据缓存机制
- 持续完善MindSpore IR
- Parameter Server模式分布式训练

## MindInsight调试调优

- 训练过程观察
    - 直方图
    - 计算图/数据图展示优化
    - 集成性能Profiling/Debugger工具
    - 支持多次训练间的对比
- 训练结果溯源
    - 数据增强溯源对比
- 训练过程诊断
    - 性能Profiling
    - 基于图模型的Debugger

## MindArmour安全增强包

- 测试模型的安全性
- 提供模型安全性增强工具
- 保护训练和推理过程中的数据隐私

## 推理框架

- 算子性能与完备度的持续优化
- 支持语音模型推理
- 端侧模型的可视化
- Micro方案，适用于嵌入式系统的超轻量化推理， 支持ARM Cortex-A、Cortex-M硬件
- 支持端侧重训及联邦学习
- 端侧自动并行特性
- 端侧MindData，包含图片Resize、像素数据转换等功能
- 配套MindSpore混合精度量化训练（或训练后量化），实现混合精度推理，提升推理性能
- 支持Kirin NPU、MTK APU等AI加速硬件
- 支持多模型推理pipeline
- C++构图接口
