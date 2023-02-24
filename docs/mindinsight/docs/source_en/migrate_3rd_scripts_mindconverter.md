# Migrating From Third Party Frameworks With MindConverter

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindinsight/docs/source_en/migrate_3rd_scripts_mindconverter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

## Overview

MindConverter is a migration tool to transform the model file of PyTorch(ONNX) or TensorFlow(PB) to MindSpore. The model file contains model structure definition(`network`) and weights information(`weights`), which will be transformed into model scripts(`model.py`) and weights file(`ckpt`) in MindSpore.

![mindconverter-overview](images/mindconverter-overview.png)

Moreover, this tool is able to transform the model file of PyTorch to MindSpore by adding API(`pytorch2mindspore`) to original PyTorch scripts.

> - Due to the strategic adjustment, MindConverter will not evolve from 1.9.0 onwards. Please note that the official website documents and codes will be gradually taken off the shelves.
> - If you are interested in the MindConverter project, please move to version 1.7.0 (refer to[MindConverter 1.7.0](https://www.mindspore.cn/mindinsight/docs/en/r1.7/migrate_3rd_scripts_mindconverter.html)).
> - MindConverter currently only maintains version 1.6.0 and 1.7.0, and the follow-up maintenance work will also gradually incline to 1.7.0.
