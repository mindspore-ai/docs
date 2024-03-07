# 迁移指南概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/migration_guide/overview.md)

本迁移指导包含以PyTorch为主的其他机器学习框架将神经网络迁移到MindSpore的完整步骤。

```{mermaid}
graph LR
A(总览)-->B(迁移流程)
B-->|Step 1|E(<font color=blue>环境准备</font>)
E-.-text1(本地安装MindSpore)
E-.-text2(在线使用ModelArts)
B-->|Step 2|F(<font color=blue>模型分析与准备</font>)
F-.-text3(算法复现/MindSpore Dev Toolkit 工具分析API满足度/分析功能满足度)
B-->|Step 3|G(<font color=blue>网络搭建对比</font>)
G-->I(<font color=blue>数据处理</font>)
I-.-text4(数据集加载/增强/读取对齐)
G-->J(<font color=blue>网络搭建</font>)
J-.-text5(网络对齐)
G-->N(<font color=blue>损失函数</font>)
N-.-text6(损失函数对齐)
G-->K(<font color=blue>学习率与优化器</font>)
K-.-text7(优化器执行和学习率策略对齐)
G-->L(<font color=blue>梯度求导</font>)
L-.-text8(反向梯度对齐)
G-->M(<font color=blue>训练及推理流程</font>)
M-.-text9(训练与推理对齐)
B-->|Step 4|H(<font color=blue>调试调优</font>)
H-.-text10(功能/精度/性能三方面对齐)
A-->C(<font color=blue>网络迁移调试实例</font>)
C-.-text11(以ReNet50为例的网络迁移样例)
A-->D(<font color=blue>FAQs</font>)
D-.-text12(常见问题与解决方法)

click C "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/sample_code.html"
click D "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/faq.html"

click E "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/enveriment_preparation.html"
click F "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/analysis_and_preparation.html"
click G "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/model_development.html"
click H "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/debug_and_tune.html"

click I "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/dataset.html"
click J "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/model_and_cell.html"
click K "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/learning_rate_and_optimizer.html"
click L "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/gradient.html"
click M "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/training_and_evaluation.html"
click N "https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/loss_function.html"
```
