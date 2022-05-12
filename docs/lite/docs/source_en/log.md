# Log

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/log.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## Log-related Environment Variables

Only server inference version support environment variables below expert GLOG_v.

- `GLOG_v`

    The environment variable specifies the log level. After the log level is specified, the log information greater than or equal to this level will be output. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.
    The default value is 2, indicating the WARNING level. ERROR level indicates that an error occurred during program execution. The error log will be output and the program may not be terminated.

- `GLOG_logtostderr`

    The environment variable specifies the log output mode.  
    When `GLOG_logtostderr` is set to 1, logs are output to the screen. If the value is set to 0, logs are output to a file. The default value is 1.

- `GLOG_log_dir`

    The environment variable specifies the log output directory path. The config path needs to exist.