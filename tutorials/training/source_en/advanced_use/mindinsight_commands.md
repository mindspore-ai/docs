# MindInsight Commands

`Linux` `Ascend` `GPU` `CPU` `Model Optimization` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/mindinsight_commands.md)

## View the Command Help Information

```shell
mindinsight --help
```

## View the Version Information

```shell
mindinsight --version
```

## Start the Service

```shell
mindinsight start [-h] [--config <CONFIG>] [--workspace <WORKSPACE>]
                  [--port <PORT>] [--url-path-prefix <URL_PATH_PREFIX>]
                  [--reload-interval <RELOAD_INTERVAL>]
                  [--summary-base-dir <SUMMARY_BASE_DIR>]
                  [--enable-debugger <ENABLE_DEBUGGER>]
                  [--debugger-port <DEBUGGER_PORT>]
```

Optional parameters as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`-h, --help`|Optional|Displays the help information about the start command.|-|-|-|-|
|`--config <CONFIG>`|Optional|Specifies the configuration file or module.|String|Empty string|-|Physical file path (file:/path/to/config.py) or a module path (python:path.to.config.module) that can be identified by Python.|
|`--workspace <WORKSPACE>`|Optional|Specifies the working directory.|String|$HOME/mindinsight|-|-|
|`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|Integer|8080|1~65535|-|
|`--url-path-prefix <URL_PATH_PREFIX>`|Optional|Specifies the URL path prefix of the web visualization service.|String|Empty string|-|URL path prefix consists of segments separated by slashes. Each segment supports alphabets / digits / underscores / dashes / dots, but not single dot or double dots.|
|`--reload-interval <RELOAD_INTERVAL>`|Optional|Specifies the interval (unit: second) for loading data.|Integer|3|-|The value 0 indicates that data is loaded only once.|
|`--summary-base-dir <SUMMARY_BASE_DIR>`|Optional|Specifies the root directory for loading training log data.|String|./|-|MindInsight traverses the direct subdirectories in this directory and searches for log files. If a direct subdirectory contains log files, it is identified as the log file directory. If a root directory contains log files, it is identified as the log file directory.|
|`--enable-debugger <ENABLE_DEBUGGER>`|Optional|Whether to launch the MindInsight Debugger.|Boolean|False|True/False|-|
|`--debugger-port <DEBUGGER_PORT>`|Optional|Specifies the port number of the debugger server.|Integer|50051|1~65535|-|

> When the service is started, the parameter values of the command line are saved as the environment variables of the process and start with `MINDINSIGHT_`, for example, `MINDINSIGHT_CONFIG`, `MINDINSIGHT_WORKSPACE`, and `MINDINSIGHT_PORT`.

## View the Service Process Information

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

## Stop the Service

```shell
mindinsight stop [-h] [--port PORT]
```

Optional parameters as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`-h, --help`|Optional|Displays the help information about the stop command.|-|-|-|-|
|`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|Integer|8080|1~65535|-|
