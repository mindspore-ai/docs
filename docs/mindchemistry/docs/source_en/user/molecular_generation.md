# Molecular Generation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindchemistry/docs/source_zh_cn/user/molecular_generation.md)

Molecular generation, using deep learning generation models to predict and generate components in the particle system. We have integrated a method based on active learning for high entropy alloy design [1], designing high entropy alloy components with extremely low thermal expansion coefficients. In the active learning process, first, candidate high entropy alloy components are generated based on AI models, and the candidate components are screened based on predictive models and thermodynamic calculations to predict the thermal expansion coefficient. Finally, researchers need to determine the final high entropy alloy components based on experimental verification.

## Supported Networks

| Function             | Model                  | Training | Inferring | Back-end   |
|:---------------------| :-------------------- | :--- | :--- |:-----------|
| molecular generation | [high_entropy_alloy_design](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/high_entropy_alloy_design)    | √    | √   | Ascend     |
