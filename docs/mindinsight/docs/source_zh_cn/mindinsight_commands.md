# MindInsight相关命令

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindinsight/docs/source_zh_cn/mindinsight_commands.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## 查看命令帮助信息

```bash
mindinsight --help
```

## 查看版本信息

```bash
mindinsight --version
```

## 启动服务

```text
mindinsight start [-h] [--workspace <WORKSPACE>] [--port <PORT>]
                  [--url-path-prefix <URL_PATH_PREFIX>]
                  [--reload-interval <RELOAD_INTERVAL>]
                  [--summary-base-dir <SUMMARY_BASE_DIR>]
                  [--enable-debugger <ENABLE_DEBUGGER>]
                  [--debugger-port <DEBUGGER_PORT>]
                  [--offline-debugger-mem-limit <OFFLINE_DEBUGGER_MEMORY_LIMIT>]
                  [--max-offline-debugger-session-num <MAX_OFFLINE_DEBUGGER_SESSION_NUMBER>]
```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`-h, --help`|可选|显示启动命令的帮助信息。|-|-|-|-|
|`--workspace <WORKSPACE>`|可选|MindInsight日志存放路径。|String|$HOME/mindinsight|-|-|
|`--port <PORT>`|可选|指定Web可视化服务端口。|Integer|8080|1~65535|-|
|`--url-path-prefix <URL_PATH_PREFIX>`|可选|指定Web服务URL地址前缀。|String|空|-|URL地址前缀由斜杠(/)分隔成多个部分，各部分支持由字母/数字/下划线/连字符/点号组成的字符串，但不能是单点号(.)或双点号(..)。|
|`--reload-interval <RELOAD_INTERVAL>`|可选|指定加载数据的时间间隔（单位：秒）。|Integer|3|0～300|设置为0时表示只加载一次数据。|
|`--summary-base-dir <SUMMARY_BASE_DIR>`|可选|指定加载训练日志数据的根目录路径。|String|./|-|MindInsight将遍历此路径下的直属子目录。若某个直属子目录包含日志文件，则该子目录被识别为日志文件目录，若根目录包含日志文件，则根目录被识别为日志文件目录。在ModelArts开发环境中，此参数可以指定为OBS路径，请参考[ModelArts文档](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0068.html)以了解更多信息。|
|`--enable-debugger <ENABLE_DEBUGGER>`|可选|是否开启Debugger功能|Boolean|False|True/False/1/0|只有开启了调试器，才会在MindInsight页面显示调试器入口。|
|`--offline-debugger-mem-limit <OFFLINE_DEBUGGER_MEMORY_LIMIT>`|可选|指定单个离线调试器会话内存使用上限（单位MB），当出现内存不足导致MindInght离线调试器运行问题时，需要用户根据内存情况设置。|Integer|16*1024|6*1024~int32上限|-|
|`--max-offline-debugger-session-num <MAX_OFFLINE_DEBUGGER_SESSION_NUMBER>`|可选|指定离线调试器会话数上限，会话数指的是能同时使用离线调试器调试的训练作业个数。|Integer|2|1~2|-|

`--workspace`日志目录说明：

| 模块名称      | 日志目录描述                                                 | 日志格式                                  |
| ------------- | ------------------------------------------------------------ | ----------------------------------------- |
| datavisual    | 训练看板模块，记录训练看板模块的所有日志                     | `datavisual.<PORT>.log`                   |
| debugger      | 调试器模块，记录调试器模块的所有日志                         | `debugger.<PORT>.log`                     |
| explainer     | 可解释AI模块，记录可解释AI模块解析数据的所有日志             | `explainer.<PORT>.log`                    |
| gunicorn      | web服务模块，记录web服务模块的所有日志                       | `access.<PORT>.log`<br>`error.<PORT>.log` |
| lineage       | 溯源模块，记录溯源模块的所有日志                             | `lineage.<PORT>.log`                      |
| notebook      | 记录在ModelArts的notebook中使用MindInsight的所有日志        | `notebook.<PORT>.log`                     |
| optimizer     | 优化器模块，记录优化器模块的所有日志                         | `optimizer.<PORT>.log`                   |
| parse_summary | summary文件解析模块，记录summary文件解析模块的所有日志 | `parse_summary.<PORT>.log`                |
| profiler      | 性能分析模块，记录性能分析模块的所有日志                     | `profiler.<PORT>.log`                     |
| restful_api   | RESTFul API模块，记录RESTFul API交互日志                      | `restful_api.<PORT>.log`                  |
| scripts       | 启停MindInsight模块，记录MindInsight启动、停止的所有日志     | `start.<PORT>.log`<br>`stop.<PORT>.log`   |
| utils         | 公共模块，记录公共模块的所有日志                             | `utils.<PORT>.log`                        |

注：每个模块一个日志文件，但单个日志文件超出50M时，文件会被重命名进行归档，被归档的文件格式为：`<module name>_<PORT>.log.<id>`。其中`module name`表示模块名，`PORT`表示端口号，`id`表示文件重命名归档次数。

> 服务启动时，命令行参数值将被保存为进程的环境变量，并以 `MINDINSIGHT_` 开头作为标识，如 `MINDINSIGHT_PORT`，`MINDINSIGHT_WORKSPACE` 等。

执行命令：

```bash
mindinsight start --port 8000 --workspace /path/to/workspace/dir --summary-base-dir /path/summary/base/dir
```

如果出现以下提示，说明启动成功：

```text
Web address: http://127.0.0.1:8000
service start state: success
```

## 查看服务进程信息

MindInsight向用户提供Web服务，可通过以下命令，查看当前运行的Web服务进程。

```bash
ps -ef | grep mindinsight
```

根据服务进程PID，可通过以下命令，查看当前服务进程对应的工作目录`WORKSPACE`。

```text
lsof -p <PID> | grep access
```

输出如下，可查看`WORKSPACE`。

```text
gunicorn  <PID>  <USER>  <FD>  <TYPE>  <DEVICE>  <SIZE/OFF>  <NODE>  <WORKSPACE>/log/gunicorn/access.log
```

## 停止服务

```text
mindinsight stop [-h] [--port PORT]
```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`-h, --help`|可选|显示停止命令的帮助信息。|-|-|-|-|
|`--port <PORT>`|可选|指定Web可视化服务端口。|Integer|8080|1~65535|-|

执行命令：

```bash
mindinsight stop --port 8000
```

如果出现以下提示，说明启动成功：

```text
Stop mindinsight service successfully
```

## Summary导出

MindInsight中提供解析Summary日志文件的工具，用户可以通过命令行将summary日志文件中的标量存入csv文件，图像存入png文件，从而便于查看和对数据进一步处理。

```text
mindinsight parse_summary [--summary-dir] [--output]
```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`--summary-dir <SUMMARY_DIR>`|可选|指定要解析的文件的目录。如果该目录中存在多个summary日志文件，则仅根据文件名解析最新的文件。|String|./|-|summary文件夹需要可读可执行权限，summary文件需要可读权限，检查权限失败会报错退出|
|`--output <OUTPUT>`|可选|指定输出的目录，将数据输出到该目录中。|String|./|-|-|

执行命令：

```bash
mindinsight parse_summary --summary-dir ./ --output ./
```

输出目录结构如下：

```text
└─output_{datetime}
    ├─image
    │   └─{tag}_{step}.png
    │
    └─scalar.csv
```

其中，

- output_{datetime}为输出目录下的新建目录，命名规则为 'output_年月日_时分秒_毫秒微秒'。

- {tag}\_{step}.png为训练过程中的图像，tag代表标签（tag中的特殊字符将被删除，'_'将被替换成代'/'）step代表训练步骤。

- scalar.csv为标量数据（编码方式：'utf-8'）。

## 使用mindoptimizer进行超参调优

MindInsight中提供调参命令，命令行（Command-line interface, CLI）的使用方式，命令如下。

```text
usage: mindoptimizer [-h] [--version] [--config <CONFIG>]
                     [--iter <ITER>]

```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`-h, --help`|可选|显示启动命令的帮助信息|-|-|-|-|
|`--config <CONFIG>`|必选|指定配置文件|String|-|-|物理文件路径（file:/path/to/config.yaml），文件格式为yaml|
|`--iter <ITER>`|可选|指定调参次数|Integer|1|正整数|-|
