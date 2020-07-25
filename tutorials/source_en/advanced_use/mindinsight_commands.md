# MindInsight Commands

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced_use/mindinsight_commands.md" target="_blank"><img src="../_static/logo_source.png"></a>

1. View the command help information.

    ```shell
    mindinsight --help
    ```

2. View the version information.

    ```shell
    mindinsight --version
    ```

3. Start the service.

    ```shell
    mindinsight start [-h] [--config <CONFIG>] [--workspace <WORKSPACE>]
                      [--port <PORT>] [--url-path-prefix <URL_PATH_PREFIX>]
                      [--reload-interval <RELOAD_INTERVAL>]
                      [--summary-base-dir <SUMMARY_BASE_DIR>]
    ```

    Optional parameters as follows:

    |Name|Argument|Description|Default|Scope|Specifications|
    |---|---|---|---|---|---|
    |`-h, --help`|Optional|Displays the help information about the startup command.|-|-|-|
    |`--config <CONFIG>`|Optional|Specifies the configuration file or module.|Empty string|-|Physical file path (file:/path/to/config.py) or a module path (python:path.to.config.module) that can be identified by Python.|
    |`--workspace <WORKSPACE>`|Optional|Specifies the working directory.|$HOME/mindinsight|-|-|
    |`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|8080|1~65535|-|
    |`--url-path-prefix <URL_PATH_PREFIX>`|Optional|Specifies the URL path prefix of the web visualization service.|Empty string|-|URL path prefix consists of segments separated by slashes. Each segment supports alphabets / digits / underscores / dashes / dots, but cannot just be emtpy string / single dot / double dots. The default value of URL_PATH_PREFIX is empty string.|
    |`--reload-interval <RELOAD_INTERVAL>`|Optional|Specifies the interval (unit: second) for loading data.|3|-|The value 0 indicates that data is loaded only once.|
    |`--summary-base-dir <SUMMARY_BASE_DIR>`|Optional|Specifies the root directory for loading training log data.|./|-|MindInsight traverses the direct subdirectories in this directory and searches for log files. If a direct subdirectory contains log files, it is identified as the log file directory. If a root directory contains log files, it is identified as the log file directory.|

    > When the service is started, the parameter values of the command line are saved as the environment variables of the process and start with `MINDINSIGHT_`, for example, `MINDINSIGHT_CONFIG`, `MINDINSIGHT_WORKSPACE`, and `MINDINSIGHT_PORT`.

4. View the service process information.

    MindInsight provides user with web services. Run the following command to view the running web service process:

    ```shell
    ps -ef | grep mindinsight
    ```

    Run the following command to access the working directory `WORKSPACE` corresponding to the service process based on the service process ID:

    ```shell
    lsof -p <PID> | grep access
    ```

    Output with the working directory `WORKSPACE` as follows:

    ```shell
    gunicorn  <PID>  <USER>  <FD>  <TYPE>  <DEVICE>  <SIZE/OFF>  <NODE>  <WORKSPACE>/log/gunicorn/access.log
    ```

5. Stop the service.

    ```shell
    mindinsight stop [-h] [--port PORT]
    ```

    Optional parameters as follows:

    |Name|Argument|Description|Default|Scope|Specifications|
    |---|---|---|---|---|---|
    |`-h, --help`|Optional|Displays the help information about the startup command.|-|-|-|
    |`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|8080|1~65535|-|
