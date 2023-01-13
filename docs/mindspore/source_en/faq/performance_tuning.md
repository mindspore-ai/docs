# Performance Tuning

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/faq/performance_tuning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `scipy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the scipy version and change the `scipy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>

> Replace version with the specific version branch of MindSpore.

<font size=3>**Q: How to choose the batchsize to achieve the best performance when training models on the Ascend chip?**</font>

A: When training the model on the Ascend chip, better training performance can be obtained when the batchsize is equal to the number of aicore or multiples. The number of Aicore can be queried via the command line in the link.
<https://support.huawei.com/enterprise/zh/doc/EDOC1100206828/eedfacda>