# Performance Tuning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/faq/performance_tuning.md)

## Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?

A: The `scipy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the scipy version and change the `scipy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/requirements.txt>

## Q: How to choose the batchsize to achieve the best performance when training models on the Ascend chip?

A: When training the model on the Ascend chip, better training performance can be obtained when the batchsize is equal to the number of AI CORE or multiples. The number of AI CORE can be queried via the command line in the link.
<https://support.huawei.com/enterprise/zh/doc/EDOC1100206828/eedfacda>
