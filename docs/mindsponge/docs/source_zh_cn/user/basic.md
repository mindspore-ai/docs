# 分子基础模型

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_zh_cn/user/basic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

分子基础模型是针对蛋白质，氨基酸或者分子做一些预训练任务，从而使下游任务取得更好的结果，如分子信息表示是AI驱动的药物设计和发现的关键先决条件，分子图预训练模型可学习到分子的丰富结构和语义信息，从而在分子性质预测等11个下游任务上与当前SOTA结果相比平均超过6%的改进。

## 已支持网络

| 功能          | 模型                            | 训练 | 推理 | 后端       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| 分子图预训练模型 | [GROVER](https://gitee.com/mindspore/mindscience/blob/f906bf284918ff2bdcd462e1c2bbf06b9af5d06a/MindSPONGE/applications/research/grover/README.md#) | ×    | √   | GPU/Ascend |