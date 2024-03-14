# Overview of Migration Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/overview.md)

This migration guide contains the complete steps for migrating neural networks to MindSpore from other machine learning frameworks, mainly PyTorch.

```{mermaid}
graph LR
A(Overview)-->B(migration process)
B-->|Step 1|E(<font color=blue>Environmental Preparation</font>)
E-.-text1(MindSpore Installation)
E-.-text2(AI Platform ModelArts)
B-->|Step 2|F(<font color=blue>Model Analysis and Preparation</font>)
F-.-text3(Reproducing algorithm, analyzing API compliance using MindSpore Dev Toolkit and analyzing function compliance.)
B-->|Step 3|G(<font color=blue>Network Constructing Comparison</font>)
G-->I(<font color=blue>Dataset</font>)
I-.-text4(Aligning the process of dataset loading, augmentation and reading)
G-->J(<font color=blue>Network Constructing</font>)
J-.-text5(Aligning the network)
G-->N(<font color=blue>Loss Function</font>)
N-.-text6(Aligning the loss function)
G-->K(<font color=blue>Learning Rate and Optimizer</font>)
K-.-text7(Aligning the optimizer and learning rate strategy)
G-->L(<font color=blue>Gradient</font>)
L-.-text8(Aligning the reverse gradients)
G-->M(<font color=blue>Training and Evaluation Process</font>)
M-.-text9(Aligning the process of training and evaluation)
B-->|Step 4|H(<font color=blue>Debug and Tuning</font>)
H-.-text10(Aligning from three aspects: function, precision and performance)
A-->C(<font color=blue>A Migration Sample</font>)
C-.-text11(The network migration sample, taking ResNet50 as an example.)
A-->D(<font color=blue>FAQs</font>)
D-.-text12(Provides the frequently-asked questions and corresponding solutions in migration process.)

click C "https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html"
click D "https://www.mindspore.cn/docs/en/master/migration_guide/faq.html"

click E "https://www.mindspore.cn/docs/en/master/migration_guide/enveriment_preparation.html"
click F "https://www.mindspore.cn/docs/en/master/migration_guide/analysis_and_preparation.html"
click G "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_development.html"
click H "https://www.mindspore.cn/docs/en/master/migration_guide/debug_and_tune.html"

click I "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/dataset.html"
click J "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/model_and_cell.html"
click K "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/learning_rate_and_optimizer.html"
click L "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/gradient.html"
click M "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/training_and_evaluation.html"
click N "https://www.mindspore.cn/docs/en/master/migration_guide/model_development/loss_function.html"
```
