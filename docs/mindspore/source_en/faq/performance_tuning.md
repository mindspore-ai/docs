# Performance Tuning

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/faq/performance_tuning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `scipy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the scipy version and change the `scipy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>

> Replace version with the specific version branch of MindSpore.

