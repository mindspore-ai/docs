# 分子基础模型

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_zh_cn/user/basic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

在生物计算、药物设计等领域，大多数任务中给训练数据打标签非常昂贵，可用于模型训练的数据集都非常小，该领域的研究者限于数据无法开发更有效的模型，导致模型精度不佳。基于生物化学与迁移学习的相关理论，分子基础模型在相关的有大量数据的任务上做预训练后，仅需使用少量数据微调即可在目标任务上得到更准确的结果。MindSPONGE提供一系列分子基础模型以及这些模型在大规模数据集上训好的checkpoint，用户可以直接在这些模型基础上根据自己的需要做精调，轻松实现高精度的模型开发。

## 已支持网络

| 功能          | 模型                            | 训练 | 推理 | 后端       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| 小分子化合物预训练模型 | [GROVER](https://gitee.com/mindspore/mindscience/pulls/441/files#) | √    | √   | GPU/Ascend |
| 小分子化合物预训练模型 | [MGBERT](https://gitee.com/mindspore/mindscience/pulls/631/files#) | √    | √   | GPU/Ascend |

后续将提供蛋白质预训练等基础模型，敬请期待。