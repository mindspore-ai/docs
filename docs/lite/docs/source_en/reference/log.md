# Log

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/reference/log.md)

## Log-related Environment Variables

Only server inference version and windows version support environment variables below, except `GLOG_v`.  

- `GLOG_v`

    The environment variable specifies the log level. After the log level is specified, the log information greater than or equal to this level will be output. The values are as follows: 0: DEBUG; 1: INFO; 2: WARNING; 3: ERROR.
    The default value is 2, indicating the WARNING level. ERROR level indicates that an error occurred during program execution. The error log will be output and the program may not be terminated.

- `GLOG_logtostderr`

    The environment variable specifies the log output mode.  
    When `GLOG_logtostderr` is set to 1, logs are output to the screen. If the value is set to 0, logs are output to a file. The default value is 1.

- `GLOG_log_dir`

    The environment variable specifies the log output directory path. The config path needs to exist.

## User-defined GLOG Options

`Mindpoint lite` supports user-defined configuration of `GLOG` related parameters for specific situations. Users can set `GLOG_USER_DEFINE` parameter to `on` to achieve custom configuration of logs. For specific GLOG options, you can use `FLAGS_**` Configure. Please refer to the table below for detailed configuration.

| Configuration item               | Definition |
|-----------------------| :----------: |
| FLAGS_log_prefix | log prefix |
| FLAGS_logbufsecs | write log to files real-time |
| FLAGS_v | log level |
| FLAGS_logfile_mode | set log file mode |
| FLAGS_max_log_size | set log file max size |
| FLAGS_logtostderr | whether to print log to screen |
