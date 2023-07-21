﻿# Performance Tuning

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/faq/source_en/performance_tuning.md)

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `SciPy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the `SciPy` version and change the `SciPy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> Replace version with the specific version branch of MindSpore.
