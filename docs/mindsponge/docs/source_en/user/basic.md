# Molecular Foundation Model

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_en/user/basic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In fields such as biological computing and drug design, it is very expensive to label training data in most tasks, and the data sets available for model training are very small. Researchers in this field cannot develop more effective models due to limited data, resulting in poor model accuracy. Based on the theories of biochemistry and transfer learning, the molecular base model can get more accurate results on the target task by using only a small amount of data fine-tuning after pre-training on the relevant task with a large amount of data. MindSPONGE provides a series of molecular foundation models and their checkpoint training based on large-scale data sets. Users can make fine-tuning directly based on these models according to their needs, enabling them to easily achieve high-precision model development.

## Supported Networks

| Function            | Model                  | Training | Inferring | Back-end       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| Molecular Compound Pre-training Model | [GROVER](https://gitee.com/mindspore/mindscience/pulls/441/files#) | √    | √   | GPU/Ascend |
| Molecular Compound Pre-training Model | [MGBERT](https://gitee.com/mindspore/mindscience/pulls/631/files#) | √    | √   | GPU/Ascend |

In the future, basic models such as protein pre-training will be provided. Please stay tuned.