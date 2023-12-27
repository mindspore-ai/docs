# Molecular Design

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_en/user/design.md)

Molecular design is an important part of drug discovery. The vast chemical space covers every possible molecule, and some virtual screening libraries now contain more than billions of molecules, but these libraries also take up only a small fraction of the chemical space. Compared with virtual screening, molecular design searches the vast chemical space to generate new molecules, but traditional experimental exploration of such a large space takes a lot of time and resources. In recent years, advances in machine learning and AI have provided new computational ideas for molecular design.

MindSpore SPONGE Biocomputing Toolkit provides a series of molecular design tools based on deep generation models to help researchers conduct efficient molecular generation research.

## Supported Networks

| Function          | Model                            | Training | Inferring | Back-end       |
| :----------- | :------------------------------ | :--- | :--- | :-------- |
| Protein Sequence Design | [ProteinMPNN](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ProteinMPNN.MD) | ×    | √   | GPU/Ascend |
| Protein Sequence Design | [ESM-IF1](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ESM-IF1.md)          | ×    | √   | GPU/Ascend |

In the future, we will also provide antibody sequence design, molecular generation and other tools. Please stay tuned.