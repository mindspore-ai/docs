# 迁移指南概述

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/overview.md)

本迁移指导包含以PyTorch为主的其他机器学习框架将神经网络迁移到MindSpore的完整步骤。

```{mermaid}
graph LR
A(总览)-->B(<font color=blue>迁移流程</font>)
B-->|Step 1|E(<font color=blue>环境准备</font>)
E-.-text1(本地安装MindSpore)
E-.-text2(在线使用ModelArts)
B-->|Step 2|F(<font color=blue>模型分析与准备</font>)
F-.-text3(使用 MindSpore Dev Toolkit 工具分析API满足度)
B-->|Step 3|G(<font color=blue>网络搭建对比</font>)
G-->I(<font color=blue>数据处理</font>)
G-->J(<font color=blue>网络搭建</font>)
G-->K(<font color=blue>学习率与优化器</font>)
G-->L(<font color=blue>梯度求导</font>)
G-->M(<font color=blue>训练及推理流程</font>)
B-->|Step 4|H(<font color=blue>调试调优</font>)
H-.-text4(从功能/精度/性能三个方面介绍一些调试调优的方法)
A-->C(<font color=blue>网络迁移调试实例</font>)
C-.-text5(以ReNet50为例的网络迁移样例)
A-->D(<font color=blue>FAQs</font>)
D-.-text6(一些常见问题与相应解决方法)

click B "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/migration_process.html"
click C "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/sample_code.html"
click D "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/faq.html"

click E "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/enveriment_preparation.html"
click F "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/analysis_and_preparation.html"
click G "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/model_development.html"
click H "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/debug_and_tune.html"

click I "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/dataset.html"
click J "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/model_and_cell.html"
click K "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/model_development/learning_rate_and_optimizer.html"
click L "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/gradient.html"
click M "https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/training_and_evaluation.html"
```
