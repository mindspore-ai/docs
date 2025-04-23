# 结构生成

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindchemistry/docs/source_zh_cn/user/structure_generation.md)

结构生成，通过深度学习的生成模型预测晶体材料的结构。我们集成了基于图神经网络和等变扩散模型的晶体生成模型 DiffCSP，它通过联合生成晶格和原子坐标来预测晶体结构，并利用周期性 E(3) 等变去噪模型来更好地模拟晶体的几何特性。它在计算成本上远低于传统的基于密度泛函理论的方法，且在晶体结构预测任务中表现出色。

## 已支持网络

| 功能            | 模型                  | 训练 | 推理 | 后端       |
| :------------- | :-------------------- | :--- | :--- | :-------- |
| 结构生成| [DiffCSP](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/diffcsp)     | √    | √   | Ascend |
