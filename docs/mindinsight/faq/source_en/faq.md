# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/faq/source_en/faq.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What can I do if the error message `ImportError: libcrypto.so.1.0.0: cannot open shared object file: No such file or directory` is displayed in the MindInsight running logs after MindInsight failed to start?**</font>

A: You can use "export LD_LIBRARY_PATH=dir:$LD_LIBRARY_PATH" command to export LD_LIBRARY_PATH variable in Linux environment.

<br/>

<font size=3>**Q: What can I do if the error message `bash: mindinsight: command not found` is displayed in the MindInsight running logs after MindInsight failed to start?**</font>

A: This problem occurs when using Python source codes to compile and install in the user-defined path. When install MindInsight by using `pip`, the executable file will be installed in this path. If the installation directory is not found in the bash environment variable queried by using 'echo $PATH', the system will not find the installed executable file. You need to use `export PATH=$PATH: $YourPythonPath$/bin` on the command line to import the path variable.

(Please change `$YourPythonPath$` to your installation path). Note: this command is only valid at the current terminal. If you want to make it permanent, please add it to the file `~/.bashrc`.

<br/>

<font size=3>**Q: What can I do if the error message `No module named 'mindinsight'` is displayed in the MindInsight running logs after MindInsight is uninstalled?**</font>

A: After MindInsight is started, it becomes a background service. After MindInsight package is uninstalled, the started MindInsight background service will not automatically stop. When the MindInsight background service starts a new process to load data or performs other operations, it will trigger the error message of `No module named 'mindinsight'` and record it to a log file.

In this case, you can perform either of the following operations:

- Reinstall MindInsight and run the `mindinsight stop --port <PORT>` command to stop the started MindInsight background service.
- Run the `kill -9 <PID>` command to kill the processes designed by MindInsight.

<br/>

<font size=3>**Q: What can I do if the Google's Chrome browser prompts the error message `ERR_UNSAFE_PORT` after MindInsight is successfully started?**</font>

A: Chrome browser's kernel prohibits certain ports from being used as HTTP services. You can add `--explicitly-allowed-ports=port` in Chrome browser's configuration. Otherwise you can change the port or browser like IE browser.

<br/>

<font size=3>**Q: What can I do if the error `Exeption calling application: Field number 0 is illegal` appears on Ascend after MindInsight is successfully started with debugger turning on, and the training script is trying to connecting to debugger?**</font>

A: It means the wrong version of Protobuf is installed, please install the right version, see [Installing Protobuf Python](https://support.huaweicloud.com/intl/en-us/instg-cli-cann/atlascli_03_0046.html).

<br/>

<font size=3>**Q: What can I do if the error `The debugger offline server module is not found` appears after MindInsight is successfully started and trying to turn on the offline debugger?**</font>

A: The debugger offline service needs to import the MindSpore. Please install the correct version of MindSpore. For the installation method, please refer to [Install MindSpore](https://www.mindspore.cn/install/en).

<br/>

<font size=3>**Q: What can I do if the Google's Chrome browser prompts the error message `ERR_CONNECTION_REFUSED` after MindInsight is successfully started?**</font>

A: Check the firewall policy configuration between the backend server and network devices to ensure that the communication between the browser and MindInsight is not restricted by the configuration rules of relative devices.
