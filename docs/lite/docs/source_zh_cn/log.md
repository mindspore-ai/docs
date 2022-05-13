# 日志

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/docs/source_zh_cn/log.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## 常用的环境变量配置

除了GLOG_v外，其它的环境变量是服务侧的推理版本才支持的。  

- `GLOG_v`

    该环境变量控制日志的级别。指定日志级别后，将会输出大于或等于该级别的日志信息，对应关系如下：0-DEBUG、1-INFO、2-WARNING、3-ERROR。  
    该环境变量默认值为2，即WARNING级别。ERROR级别表示程序执行出现报错，输出错误日志。  

- `GLOG_logtostderr`

    该环境变量控制日志的输出方式。  
    该环境变量的值设置为1时，日志输出到屏幕；值设置为0时，日志输出到文件。默认值为1。  

- `GLOG_log_dir`

    该环境变量指定日志输出的路径，用户须确保指定的路径真实存在。  
