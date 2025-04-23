# 分子预测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindchemistry/docs/source_zh_cn/user/molecular_prediction.md)

分子性质预测，通过深度学习网络预测不同粒子体系中的各种性质. 我们集成了NequIP模型、Allegro模型，根据分子体系中各原子的位置与原子数信息构建图结构描述，基于等变计算与图神经网络，计算出分子体系能量。
密度泛函理论哈密顿量预测。我们集成了DeephE3nn模型，基于E3的等变神经网络，利用原子的结构去预测其的哈密顿量。
晶体材料性质预测。我们集成了Matformer模型，基于图神经网络和Transformer架构的模型，用于预测晶体材料的各种性质。

## 已支持网络

| 功能            | 模型                                                                                                              | 训练 | 推理 | 后端       |
| :------------- |:----------------------------------------------------------------------------------------------------------------| :--- | :--- | :-------- |
| 分子预测| [Nequip](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/nequip)    | √    | √   | Ascend |
| 分子预测| [Allgro](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/allegro)    | √    | √   | Ascend |
| 电子结构预测| [Deephe3nn](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/deephe3nn) | √    | √   | Ascend |
| 晶体材料性质预测| [Matformer](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/matformer) | √    | √   | Ascend |