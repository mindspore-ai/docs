﻿# Performance Tuning

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_en/performance_tuning.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `SciPy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the `SciPy` version and change the `SciPy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> Replace version with the specific version branch of MindSpore.
