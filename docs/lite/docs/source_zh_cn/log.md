# 日志

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/log.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 常用的环境变量配置

- `GLOG_v`

    该环境变量控制日志的级别。指定日志级别后，将会输出大于或等于该级别的日志信息，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR。
    该环境变量默认值为2，即WARNING级别。ERROR级别表示程序执行出现报错，输出错误日志。

- `GLOG_logtostderr`

    该环境变量控制日志的输出方式。  
    该环境变量的值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。

- `GLOG_log_dir`

    该环境变量指定日志输出的路径，日志保存路径为：`指定的路径/rank_${rank_id}/logs/`。非分布式训练场景下，`rank_id`为0；分布式训练场景下，`rank_id`为当前设备在集群中的ID。  
    若`GLOG_logtostderr`的值为0，则必须设置此变量。  
    若指定了`GLOG_log_dir`且`GLOG_logtostderr`的值为1时，则日志输出到屏幕，不输出到文件。  
    C++和Python的日志会被输出到不同的文件中，C++日志的文件名遵从`GLOG`日志文件的命名规则，这里是`mindspore.机器名.用户名.log.日志级别.时间戳.进程ID`，Python日志的文件名为`mindspore.log.进程ID`。  
    `GLOG_log_dir`只能包含大小写字母、数字、"-"、"_"、"/"等字符。
