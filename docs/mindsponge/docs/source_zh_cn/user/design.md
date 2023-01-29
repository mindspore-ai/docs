# 分子设计

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_zh_cn/user/design.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

分子设计是药物发现的重要组成部分。广阔的化学空间涵盖了所有可能的分子，目前有些虚拟筛选库已经包含超过数十亿个分子，但是这些库也只占化学空间的很小一部分。与虚拟筛选相比，分子设计从广阔的化学空间搜索生成新的分子，但是传统的实验探索如此大的空间需要花费大量的时间和资源。近年来由于机器学习和AI方法的进步，为分子设计提供了新的计算思路。

MindSPONGE生物计算工具包提供一系列基于深度生成模型的分子设计工具，帮助研究者进行高效的分子生成研究。

## 已支持网络

| 功能          | 模型                            | 训练 | 推理 | 后端       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| 蛋白质序列设计 | [ProteinMPNN](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/ProteinMPNN/README.md#) | ×    | √   | GPU/Ascend |
| 蛋白质序列设计 | [ESM-IF1](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/esm/README.md#)          | ×    | √   | GPU/Ascend |

未来我们还将提供抗体序列设计，分子生成等工具，敬请期待。