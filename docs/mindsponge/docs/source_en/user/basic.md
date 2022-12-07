# Molecular Basic Model

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_en/user/basic.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Molecular basic model is to perform some pre-training tasks for proteins, amino acids or molecules, so as to achieve better results for downstream tasks. For example, molecular information representation is a key prerequisite for AI-driven drug design and discovery. Molecular diagram pre-training model can learn the rich structure and semantic information of molecules. This results in an average improvement of more than 6% over the current SOTA results for 11 downstream tasks such as molecular property prediction.

## Supported Networks

| Function            | Model                  | Training | Inferring | Back-end       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| Molecular Graph Pre-training Model | [GROVER](https://gitee.com/mindspore/mindscience/blob/f906bf284918ff2bdcd462e1c2bbf06b9af5d06a/MindSPONGE/applications/research/grover/README.md#) | ×    | √   | GPU/Ascend |