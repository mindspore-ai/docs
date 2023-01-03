# Molecular Structure Prediction

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_en/user/structure_prediction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The acquisition of molecular structure, especially the structure of biomacromolecules (DNA, RNA and protein), is an important issue in the field of biopharmaceutical research and has a wide range of uses. The MindSPONGE biological computing toolkit provides a series of calculation tools for molecular structure prediction, helping researchers to acquire high-precision molecular structure efficiently.

Currently, a series of tools for protein and RNA structure prediction are available, which support high precision prediction of protein monomer and complex structure and RNA secondary structure.

## Supported Networks

| Function       | Model                                        | Training | Inferring | Back-end   |
| :------------- | :------------------------------------------- | :--- | :--- | :-------- |
| Single Chain Structure Prediction | [MEGA-Fold](https://gitee.com/mindspore/mindscience/blob/r2.0.0-alpha/MindSPONGE/applications/MEGAProtein/README.md#)                   | √    | √   | GPU/Ascend |
| MSA Generation/Correction    | [MEGA-EvoGen](https://gitee.com/mindspore/mindscience/blob/r2.0.0-alpha/MindSPONGE/applications/MEGAProtein/README.md#)               | √    | √   | GPU/Ascend |
| Structural Quality Assessment | [MEGA-Assessment](https://gitee.com/mindspore/mindscience/blob/r2.0.0-alpha/MindSPONGE/applications/MEGAProtein/README.md#)       | √    | √   | GPU/Ascend |
| Multi-chain Structure Prediction | [AlphaFold-Multimer](https://gitee.com/mindspore/mindscience/blob/r2.0.0-alpha/MindSPONGE/applications/research/Multimer/README.md#) | ×    | √   | GPU/Ascend |
| RNA Secondary Structure Prediction | [UFold](https://gitee.com/mindspore/mindscience/blob/r2.0.0-alpha/MindSPONGE/applications/research/UFold/README.md#)                          | √    | √   | GPU/Ascend |

In the future, we will further improve the function of molecular structure prediction, and introduce more tools for protein-ligand complex structure prediction and small molecule structure prediction of compounds. Stay tuned for more information.