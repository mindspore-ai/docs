# Molecular Prediction

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindchemistry/docs/source_zh_cn/user/molecular_prediction.md)

Molecular property prediction, predicting various properties in different particle systems through deep learning networks. We integrated the NequIP model [2] and Allegro model [3] to construct a graph structure description based on the position and number of atoms in the molecular system. Using equivariant calculations and graph neural networks, we calculated the energy of the molecular system.
Density Functional Theory Hamiltonian Prediction. We integrate the DeephE3nn model, an equivariant neural network based on E3, to predict a Hamiltonian by using the structure of atoms.
Prediction of crystalline material properties. We integrate the Matformer model based on graph neural networks and Transformer architectures, for predicting various properties of crystalline materials.

## Supported Networks

| Function                        | Model                                                                                                 | Training | Inferring | Back-end |
|:--------------------------------|:------------------------------------------------------------------------------------------------------| :--- | :--- |:---------|
| Property prediction             | [Nequip](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/nequip)       | √    | √   | Ascend   |
| Property prediction             | [Allgro](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/allegro)      | √    | √   | Ascend   |
| Electronic structure prediction | [Deephe3nn](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/deephe3nn) | √    | √   | Ascend   |
| Prediction of crystalline material properties                        | [Matformer](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/matformer) | √    | √   | Ascend   |