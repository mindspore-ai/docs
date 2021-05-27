# Performance Tuning

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<!-- TOC -->

- [Performance Tuning](#performance-tuning)
    - [Parameter Tuning](#parameter-tuning)
    - [MindInsight Using](#mindinsight-using)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_cn/performance_tuning.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Performance Tuning

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `SciPy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the `SciPy` version and change the `SciPy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> Replace version with the specific version branch of MindSpore.

## MindInsight Using

<font size=3>**Q: What can I do if the error message `ImportError: libcrypto.so.1.0.0: cannot open shared object file: No such file or directory` is displayed in the MindInsight running logs after MindInsight failed to start?**</font>

A: You can use "export LD_LIBRARY_PATH=dir:$LD_LIBRARY_PATH" command to export LD_LIBRARY_PATH variable in Linux environment.

<br />

<font size=3>**Q: What can I do if the error message `No module named 'mindinsight'` is displayed in the MindInsight running logs after MindInsight is uninstalled?**</font>

A: After MindInsight is started, it becomes a background service. After MindInsight package is uninstalled, the started MindInsight background service will not automatically stop. When the MindInsight background service starts a new process to load data or performs other operations, it will trigger the error message of `No module named 'mindinsight'` and record it to a log file.

In this case, you can perform either of the following operations:

- Reinstall MindInsight and run the `mindinsight stop --port <PORT>` command to stop the started MindInsight background service.
- Run the `kill -9 <PID>` command to kill the processes designed by MindInsight.

<br />

<font size=3>**Q: What can I do if the Google's Chrome browser prompts the error message `ERR_UNSAFE_PORT after` MindInsight is successfully started?**</font>

A: Chrome browser's kernel prohibits certain ports from being used as HTTP services. You can add `--explicitly-allowed-ports=port` in Chrome browser's configuration. Otherwise you can change the port or browser like IE browser.

<br />

<font size=3>**Q: What can I do if the error `Exeption calling application: Field number 0 is illegal` appears on Ascend after MindInsight is successfully started with debugger turning on, and the training script is trying to connecting to debugger?**</font>

A: It means the wrong version of protobuf is installed, please install the right version, see [Installing protobuf Python](https://support.huaweicloud.com/intl/en-us/instg-cli-cann/atlascli_03_0046.html).
