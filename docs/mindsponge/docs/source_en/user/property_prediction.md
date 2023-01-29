# Molecular Properties Prediction

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_en/user/property_prediction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

Molecular property prediction is one of the most important tasks in the computer-aided drug discovery process and plays an important role in many downstream applications such as drug screening and drug design. Density Functional Theory (DFT) is mostly used in traditional molecular property prediction. Although DFT can accurately predict a variety of molecular properties, the calculation is very time-consuming, often requiring several hours to complete the property calculation of a single molecule. In addition, the number of candidate compounds is relatively large, so using traditional quantum chemistry methods to predict molecular properties needs to pay huge resources and time costs. Thanks to the rapid development of deep learning, more and more people begin to try to apply deep learning to the field of molecular property prediction. Its main purpose is to predict molecular physical and chemical properties through internal molecular information such as atomic coordinates and atomic numbers, so as to help people quickly find compounds that meet the predicted properties among a large number of candidate compounds, and speed up drug screening and drug design.

## Supported Networks

| Function            | Model                  | Training | Inferring | Back-end       |
| :------------- | :-------------------- | :--- | :--- | :-------- |
| Drug Interaction Prediction | [KGNN](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/KGNN/README.md#)     | √    | √   | GPU/Ascend |
| Drug Disease Association Prediction | [DeepDR](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/DeepDR/README.en.md#) | √    | √   | GPU/Ascend |
| Protein-Ligand Affinity Prediction | [pafnucy](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindSPONGE/applications/research/pafnucy/README.md#) | √   | √   | GPU/Ascend |

Molecular docking, ADMET and other molecular property prediction networks will be provided in the future, please stay tuned.