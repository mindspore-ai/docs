# 分子性质预测

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_zh_cn/user/property_prediction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

分子性质预测是计算机辅助药物发现流程中最重要的任务之一，在许多下游应用（例如药物筛选，药物设计）中都发挥着重要作用。传统分子性质预测使用密度泛函理论（Density Functional Theory, DFT）进行计算居多，虽然DFT可以精准预测多种分子性质，然而计算非常耗时，往往需要数个小时才能完成单个分子的性质计算。此外候选化合物数量较为庞大，因此使用传统量子化学方法进行分子性质预测需要付出巨大的资源和时间成本。得益于深度学习的快速发展，越来越多的人们开始尝试将深度学习应用于分子性质预测这一领域。其主要目的是通过原子坐标、原子序数等分子内部信息，对分子物理、化学性质做出预测，从而帮助人们快速在大量候选化合物中找到符合预测性质的化合物，加快药物筛选和药物设计的速度。

## 已支持网络

| 功能            | 模型                  | 训练 | 推理 | 后端       |
| :------------- | :-------------------- | :--- | :--- | :-------- |
| 药物相互作用预测 | [KGNN](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/research/KGNN/README.md#)     | √    | √   | GPU/Ascend |
| 药物疾病关联预测 | [DeepDR](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/research/DeepDR/README.md#) | √    | √   | GPU/Ascend |
| 蛋白质-配体亲和力预测 | [pafnucy](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/research/pafnucy/README.md#) | √   | √   | GPU/Ascend |

后续将提供分子对接，ADMET等分子性质预测网络，敬请期待。