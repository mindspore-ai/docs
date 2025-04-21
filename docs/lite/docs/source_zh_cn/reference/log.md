# 日志

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/reference/log.md)

## 常用的环境变量配置

除了GLOG_v外，其他的环境变量是服务侧的推理版本以及windows版本才支持的。  

- `GLOG_v`

    该环境变量控制日志的级别。指定日志级别后，将会输出大于或等于该级别的日志信息，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR。  
    该环境变量默认值为2，即WARNING级别。ERROR级别表示程序执行出现报错，输出错误日志。  

- `GLOG_logtostderr`

    该环境变量控制日志的输出方式。  
    该环境变量的值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。  

- `GLOG_log_dir`

    该环境变量指定日志输出的路径，用户须确保指定的路径真实存在。

## 用户自定义设置GLOG选项

针对特定情况，`mindspore lite`支持用户自定义配置GLOG相关参数。用户可以通过设置`GLOG_USER_DEFINE`参数为`on`，实现日志的自定义配置。针对特定的GLOG选项，可以通过`FLAGS_**`进行配置，详细配置见下表格。

| 配置项               | 含义 |
|-----------------------| :----------: |
| FLAGS_log_prefix | 日志前缀 |
| FLAGS_logbufsecs | 日志实时写入文件 |
| FLAGS_v | 日志级别 |
| FLAGS_logfile_mode | 日志文件权限模式 |
| FLAGS_max_log_size | 日志文件大小 |
| FLAGS_logtostderr | 日志是否打印到屏幕 |
