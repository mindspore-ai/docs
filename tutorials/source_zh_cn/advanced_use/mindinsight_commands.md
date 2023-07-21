# MindInsight相关命令

`Ascend` `GPU` `CPU` `模型调优` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_zh_cn/advanced_use/mindinsight_commands.md)

## 查看命令帮助信息

```shell
mindinsight --help
```

## 查看版本信息

```shell
mindinsight --version
```

## 启动服务

```shell
mindinsight start [-h] [--config <CONFIG>] [--workspace <WORKSPACE>]
                  [--port <PORT>] [--url-path-prefix <URL_PATH_PREFIX>]
                  [--reload-interval <RELOAD_INTERVAL>]
                   [--summary-base-dir <SUMMARY_BASE_DIR>]
```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`-h, --help`|可选|显示启动命令的帮助信息。|-|-|-|-|
|`--config <CONFIG>`|可选|指定配置文件或配置模块。|String|空|-|物理文件路径（file:/path/to/config.py）或Python可识别的模块路径（python:path.to.config.module）。|
|`--workspace <WORKSPACE>`|可选|指定工作目录路径。|String|$HOME/mindinsight|-|-|
|`--port <PORT>`|可选|指定Web可视化服务端口。|Integer|8080|1~65535|-|
|`--url-path-prefix <URL_PATH_PREFIX>`|可选|指定Web服务URL地址前缀。|String|空|-|URL地址前缀由斜杠(/)分隔成多个部分，各部分支持由字母/数字/下划线/连字符/点号组成的字符串，但不能是单点号(.)或双点号(..)。|
|`--reload-interval <RELOAD_INTERVAL>`|可选|指定加载数据的时间间隔（单位：秒）。|Integer|3|-|设置为0时表示只加载一次数据。|
|`--summary-base-dir <SUMMARY_BASE_DIR>`|可选|指定加载训练日志数据的根目录路径。|String|./|-|MindInsight将遍历此路径下的直属子目录。若某个直属子目录包含日志文件，则该子目录被识别为日志文件目录，若根目录包含日志文件，则根目录被识别为日志文件目录。|

> 服务启动时，命令行参数值将被保存为进程的环境变量，并以 `MINDINSIGHT_` 开头作为标识，如 `MINDINSIGHT_CONFIG`，`MINDINSIGHT_WORKSPACE`，`MINDINSIGHT_PORT` 等。

## 查看服务进程信息

MindInsight向用户提供Web服务，可通过以下命令，查看当前运行的Web服务进程。

```shell
ps -ef | grep mindinsight
```

根据服务进程PID，可通过以下命令，查看当前服务进程对应的工作目录`WORKSPACE`。

```shell
lsof -p <PID> | grep access
```

输出如下，可查看`WORKSPACE`。

```shell
gunicorn  <PID>  <USER>  <FD>  <TYPE>  <DEVICE>  <SIZE/OFF>  <NODE>  <WORKSPACE>/log/gunicorn/access.log
```

## 停止服务

```shell
mindinsight stop [-h] [--port PORT]
```

参数含义如下:

|参数名|属性|功能描述|参数类型|默认值|取值范围|规则限制|
|---|---|---|---|---|---|---|
|`-h, --help`|可选|显示停止命令的帮助信息。|-|-|-|-|
|`--port <PORT>`|可选|指定Web可视化服务端口。|Integer|8080|1~65535|-|
