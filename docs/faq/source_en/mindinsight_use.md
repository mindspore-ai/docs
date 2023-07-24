# MindInsight use

`Linux` `Ascend` `GPU` `Environment Preparation`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_en/mindinsight_use.md)

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