# Overview of Migration Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindspore/source_en/migration_guide/overview.md)

This migration guide contains the complete steps for migrating neural networks to MindSpore from other machine learning frameworks, mainly PyTorch.

```{mermaid}
flowchart LR
    A(Overview)-->B(migration process)
    B-->|Step 1|E(<font color=blue>Environmental Preparation</font>)
    E-.-text1(MindSpore Installation)
    E-.-text2(AI Platform ModelArts)
    B-->|Step 2|F(<font color=blue>Model Analysis and Preparation</font>)
    F-.-text3("Reproducing algorithm,
    analyzing API compliance
    using MindSpore Dev Toolkit
    and analyzing function compliance.")
    B-->|Step 3|G(<font color=blue>Network Constructing Comparison</font>)
    G-->I(<font color=blue>Dataset</font>)
    I-.-text4("Aligning the process of
    dataset loading,
    augmentation and reading")
    G-->J(<font color=blue>Network Constructing</font>)
    J-.-text5(Aligning the network)
    G-->K(<font color=blue>Loss Function</font>)
    K-.-text6(Aligning the loss function)
    G-->L(<font color=blue>Learning Rate and Optimizer</font>)
    L-.-text7("Aligning the optimizer
    and learning rate strategy")
    G-->M(<font color=blue>Gradient</font>)
    M-.-text8("Aligning the reverse
    gradients")
    G-->N(<font color=blue>Training and Evaluation Process</font>)
    N-.-text9("Aligning the process of
    training and evaluation")
    B-->|Step 4|H(<font color=blue>Debug and Tune</font>)
    H-->O(<font color=blue>Function Debugging</font>)
    O-.-text10(Functional alignment)
    H-->P(<font color=blue>Precision Tuning</font>)
    P-.-text11(Precision alignment)
    H-->Q(<font color=blue>Performance Tuning</font>)
    Q-.-text12(Performance Alignment)
    A-->C(<font color=blue>A Migration Sample</font>)
    C-.-text13("The network migration
    sample, taking ResNet50 as an example.")
    A-->D(<font color=blue>Reference</font>)
    D-->R(<font color=blue>PyTorch and MindSpore API Mapping Table</font>)
    D-->S(<font color=blue>Application Practice Guide for Network Migration Tool</font>)
    D-->T(<font color=blue>FAQs</font>)

    click C "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/sample_code.html"
    click D "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/reference.html"

    click E "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/enveriment_preparation.html"
    click F "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/analysis_and_preparation.html"
    click G "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/model_development.html"
    click H "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/debug_and_tune.html#debug-and-tune"

    click I "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/dataset.html"
    click J "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/model_and_cell.html"
    click K "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/loss_function.html"
    click L "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/learning_rate_and_optimizer.html"
    click M "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/gradient.html"
    click N "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/model_development/training_and_evaluation.html"

    click O "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/debug_and_tune.html#function-debugging"
    click P "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/debug_and_tune.html#precision-tuning"
    click Q "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/debug_and_tune.html#performance-tuning"

    click R "https://www.mindspore.cn/docs/en/r2.4.10/note/api_mapping/pytorch_api_mapping.html"
    click S "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/migrator_with_tools.html"
    click T "https://www.mindspore.cn/docs/en/r2.4.10/migration_guide/faq.html"
```
