# Structure Generation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindchemistry/docs/source_en/user/structure_generation.md)

Structure generation, which is a structure generation model based on deep learning to predict the structures of crystalline materials. DiffCSP integrates graph neural networks and equivalent diffusion models to jointly generate crystal lattices and atomic coordinates. It also leverages a periodic E(3)-equivalent denouncing model to better simulate the geometric properties of crystals. Compared with traditional methods based on density functional theory, DiffCSP significantly reduces computational costs and demonstrates excellent performance in crystal structure prediction tasks.

## Supported Networks

| Function             | Model                  | Training | Inferring | Back-end   |
|:---------------------| :-------------------- | :--- | :--- |:-----------|
| structure generation | [DiffCSP](https://gitee.com/mindspore/mindscience/tree/r0.7/MindChemistry/applications/diffcsp)    | √    | √   | Ascend     |
