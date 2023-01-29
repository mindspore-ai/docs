# 分子结构预测

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_zh_cn/user/structure_prediction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

获取分子结构，尤其是生物大分子（DNA、RNA、蛋白质）的结构，是生物制药领域研究的重要问题，其用途也十分广泛，MindSPONGE生物计算工具包提供一系列分子结构预测的计算工具，帮助研究者高效获取高精度的分子结构。

当前已开放了蛋白质与RNA结构预测的一系列工具，支持高精度预测蛋白质单体与复合物结构以及RNA的二级结构。

## 已支持网络

| 功能            | 模型                                         | 训练 | 推理 | 后端       |
| :------------- | :------------------------------------------- | :--- | :--- | :-------- |
| 单链结构预测    | [MEGA-Fold](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/MEGAProtein/README_CN.md#)                   | √    | √   | GPU/Ascend |
| MSA生成/修正    | [MEGA-EvoGen](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/MEGAProtein/README_CN.md#)               | √    | √   | GPU/Ascend |
| 结构质量评估    | [MEGA-Assessment](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/MEGAProtein/README_CN.md#)       | √    | √   | GPU/Ascend |
| 多链结构预测    | [AlphaFold-Multimer](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/Multimer/README.md#) | ×    | √   | GPU/Ascend |
| RNA二级结构预测 | [UFold](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/UFold/README_CN.md#)                          | √    | √   | GPU/Ascend |

未来我们将进一步完善分子结构预测的相关功能，推出蛋白质-配体复合结构预测以及化合物小分子结构预测等更多工具，敬请期待。