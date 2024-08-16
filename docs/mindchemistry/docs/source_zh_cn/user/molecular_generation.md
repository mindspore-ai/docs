# 分子生成

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindchemistry/docs/source_zh_cn/user/molecular_generation.md)

分子生成，通过深度学习的生成模型去预测并生成粒子体系中的组成. 我们集成了基于主动学习进行高熵合金设计的方法，设计热膨胀系数极低的高熵合金组分。在主动学习流程中，首先基于AI模型生成候选的高熵合金组分，并基于预测模型和热动力学计算预测热膨胀系数对候选组分进行筛选，最终需要研究者基于实验验证确定最终的高熵合金组分。

## 已支持网络

| 功能            | 模型                  | 训练 | 推理 | 后端       |
| :------------- | :-------------------- | :--- | :--- | :-------- |
| 分子生成| [high_entropy_alloy_design](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/high_entropy_alloy_design)     | √    | √   | Ascend |
