# MindInsight Commands

`Linux` `Ascend` `GPU` `CPU` `Model Optimization` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindinsight/docs/source_en/mindinsight_commands.md)

## View the Command Help Information

```bash
mindinsight --help
```

## View the Version Information

```bash
mindinsight --version
```

## Start the Service

```text
mindinsight start [-h] [--workspace <WORKSPACE>] [--port <PORT>]
                  [--url-path-prefix <URL_PATH_PREFIX>]
                  [--reload-interval <RELOAD_INTERVAL>]
                  [--summary-base-dir <SUMMARY_BASE_DIR>]
                  [--enable-debugger <ENABLE_DEBUGGER>]
                  [--debugger-port <DEBUGGER_PORT>]
```

Optional parameters are as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`-h, --help`|Optional|Displays the help information about the start command.|-|-|-|-|
|`--workspace <WORKSPACE>`|Optional|Specifies the working directory.|String|$HOME/mindinsight|-|-|
|`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|Integer|8080|1~65535|-|
|`--url-path-prefix <URL_PATH_PREFIX>`|Optional|Specifies the URL path prefix of the web visualization service.|String|Empty string|-|URL path prefix consists of segments separated by slashes. Each segment supports alphabets / digits / underscores / dashes / dots, but not single dot or double dots.|
|`--reload-interval <RELOAD_INTERVAL>`|Optional|Specifies the interval (unit: second) for loading data.|Integer|3|0~300|The value 0 indicates that data is loaded only once.|
|`--summary-base-dir <SUMMARY_BASE_DIR>`|Optional|Specifies the root directory for loading training log data.|String|./|-|MindInsight traverses the direct subdirectories in this directory and searches for log files. If a direct subdirectory contains log files, it is identified as the log file directory. If a root directory contains log files, it is identified as the log file directory.|
|`--enable-debugger <ENABLE_DEBUGGER>`|Optional|Whether to launch the MindInsight Debugger.|Boolean|False|True/False/1/0|The debugger entry can be shown on MindInsight UI only when MindInsight Debugger is launched.|
|`--debugger-port <DEBUGGER_PORT>`|Optional|Specifies the port number of the debugger server.|Integer|50051|1~65535|-|

> When the service is started, the parameter values of the command line are saved as the environment variables of the process and start with `MINDINSIGHT_`, for example, `MINDINSIGHT_PORT`, `MINDINSIGHT_WORKSPACE`, etc.

Execute command:

```bash
mindinsight start --port 8000 --workspace /path/to/workspace/dir --summary-base-dir /path/to/summary/base/dir
```

The startup is successful if it prompts as follows:

```text
Web address: http://127.0.0.1:8000
service start state: success
```

## View the Service Process Information

MindInsight provides user with web services. Run the following command to view the running web service process:

```bash
ps -ef | grep mindinsight
```

Run the following command to access the working directory `WORKSPACE` corresponding to the service process based on the service process ID:

```text
lsof -p <PID> | grep access
```

Output the working directory `WORKSPACE` as follows:

```text
gunicorn  <PID>  <USER>  <FD>  <TYPE>  <DEVICE>  <SIZE/OFF>  <NODE>  <WORKSPACE>/log/gunicorn/access.log
```

## Stop the Service

```text
mindinsight stop [-h] [--port PORT]
```

Optional parameters are as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`-h, --help`|Optional|Displays the help information about the stop command.|-|-|-|-|
|`--port <PORT>`|Optional|Specifies the port number of the web visualization service.|Integer|8080|1~65535|-|

Execute command:

```bash
mindinsight stop --port 8000
```

The shutdown is successful if it prompts as follows:

```text
Stop mindinsight service successfully
```

## Parse summary

MindInsight provides tools for parsing summary log files. Users can save the scalars in the summary log file into a csv file and the images into a png file through the commands, which is convenient for viewing and further processing.

```text
mindinsight parse_summary [--summary-dir] [--output]
```

Optional parameters are as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`--summary-dir`|Optional|Specifies the root directory of summary files. If the directory contains multiple summary files, only the latest summary file is parsed.|String|./|-|The summary file directory needs to be readable and executable, and the summary file needs to be readable.|
|`--output`|Optional|Specifies the root directory for saving output files.|String|./|-|-|

Execute command:

```bash
mindinsight parse_summary --summary-dir ./ --output ./
```

The output directory structure is as follows:

```text
└─output_{datetime}
    ├─image
    │   └─{tag}_{step}.png
    │
    └─scalar.csv
```

In which,

- output_{datetime} is the output directory. The rule is 'output_yyyyMMdd_HHmmss_SSSSSS' including year, month, day, hour, minute, second and microseconds.

- {tag}\_{step}.png is the image in training process. 'tag' and 'step' are the tag and step in the training (special characters in tag are deleted and '/' is replaced by '_').

- scalar.csv is the file which save scalars (encoding: 'utf-8').

## Use Mindoptimizer to Tune Hyperparameters

MindInsight provides parameters tuning command. The command-line interface (CLI) provides the following commands:

```text
usage: mindoptimizer [-h] [--version] [--config <CONFIG>]
                     [--iter <ITER>]

```

Optional parameters are as follows:

|Name|Argument|Description|Type|Default|Scope|Specifications|
|---|---|---|---|---|---|---|
|`-h, --help`|Optional|Displays the help information about the start command.|-|-|-|-|
|`--config <CONFIG>`|Required|Specifies the configuration file.|String|-|-|Physical file path (file:/path/to/config.yaml), and the file format is yaml.|
|`--iter <ITER>`|Optional|Specifies the run times for tuning parameters|Integer|1|Positive integer|-|
