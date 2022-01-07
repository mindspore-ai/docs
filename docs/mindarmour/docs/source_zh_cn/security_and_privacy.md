# MindArmour模块介绍

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/security_and_privacy.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

本篇主要介绍AI安全与隐私保护。AI作为一种通用技术，在带来巨大机遇和效益的同时也面临着新的安全与隐私保护的挑战。MindArmour是MindSpore的一个子项目，为MindSpore提供安全与隐私保护能力，主要包括对抗鲁棒性、模型安全测试、差分隐私训练、隐私泄露风险评估等技术。

## 对抗鲁棒性

### Attack

`Attack`基类定义了对抗样本生成的使用接口，其子类实现了各种具体的生成算法，支持安全工作人员快速高效地生成对抗样本，用于攻击AI模型，以评估模型的鲁棒性。

### Defense

`Defense`基类定义了对抗训练的使用接口，其子类实现了各种具体的对抗训练算法，增强模型的对抗鲁棒性。

### Detector

`Detector`基类定义了对抗样本检测的使用接口，其子类实现了各种具体的检测算法，增强模型的对抗鲁棒性。

详细内容，请参考[对抗鲁棒性官网教程](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/improve_model_security_nad.html)。

## 模型安全测试

### Fuzzer

`Fuzzer`类基于神经元覆盖率增益控制fuzzing流程，采用自然扰动和对抗样本生成方法作为变异策略，激活更多的神经元，从而探索不同类型的模型输出结果、错误行为，指导用户增强模型鲁棒性。

详细内容，请参考[模型安全测试官网教程](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/test_model_security_fuzzing.html)。

## 差分隐私训练

### DPModel

`DPModel`继承了`mindspore.Model`，提供了差分隐私训练的入口函数。

详细内容，请参考[差分隐私官网教程](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_differential_privacy.html)。

## 抑制隐私训练

### SuppressModel

`SuppressModel`继承了`mindspore.Model`，提供了抑制隐私训练的入口函数。

详细内容，请参考[抑制隐私官网教程](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html)。

## 隐私泄露风险评估

### MembershipInference

`MembershipInference`类提供了一种模型逆向分析方法，能够基于模型对样本的预测信息，推测某个样本是否在模型的训练集中，以此评估模型的隐私泄露风险。

详细内容，请参考[隐私泄露风险评估官方教程](https://www.mindspore.cn/mindarmour/docs/zh-CN/master/test_model_security_membership_inference.html)。
