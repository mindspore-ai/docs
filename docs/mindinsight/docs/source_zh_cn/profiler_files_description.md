# profiler目录下的性能数据

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_zh_cn/profiler_files_description.md)

profiler目录下包括一些csv、json、txt文件，这些文件包含了模型计算过程中算子执行时间、内存占用、通信等性能数据，帮助用户分析性能瓶颈。下面对部分csv、txt文件中的字段进行说明，文件内容主要包括device侧算子（AI Core算子和AI CPU算子）耗时的信息、算子级内存和应用级内存占用的信息。

## aicore_intermediate_*_detail.csv文件说明

aicore_intermediate_\*_detail.csv文件包含基于output_timeline_data_\*.txt和framework_raw_\*.csv中的内容，统计AI Core算子信息。文件中的字段说明参考下表：

| 字段名                     | 字段说明 |
|----------------------------|----------------------------|
|full_kernel_name            |device侧执行kernel算子全名|
|task_duration               |算子执行用时|
|execution_frequency         |算子执行频次|
|task_type                   |算子的任务类型|

## aicore_intermediate_*_type.csv文件说明

aicore_intermediate_\*_type.csv文件包括基于output_timeline_data_\*.txt和framework_raw_\*.csv中的内容，统计AI Core算子具体类型的信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  kernel_type                 | AI Core算子类型|
|  task_time                   | 该类型算子总用时|
|  execution_frequency         | 该类型算子执行频次|
|  percent                     | 该算子类型的用时的占所有算子总用时的百分比|

## aicpu_intermediate_*.csv文件说明

aicpu_intermediate_\*.csv文件包含AI CPU算子的耗时信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  serial_num                  | AI CPU算子序号|
|  kernel_type                 | AI CPU算子类型|
|  total_time                  | 算子耗时，等于下发耗时和执行耗时之和|
|  dispatch_time               | 下发耗时|
|  execution_time              | 执行耗时|
|  run_start                   | 算子执行起始时间|
|  run_end                     | 算子执行结束时间|

## flops_*.txt文件说明

flops_\*.txt文件包含device侧算子的浮点计算次数、每秒浮点计算次数等信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  full_kernel_name            | device侧执行kernel算子全名|
|  MFLOPs(10^6 cube)           | 浮点计算次数(10^6 cube)|
|  GFLOPS(10^9 cube)           | 每秒浮点计算次数(10^9 cube)|
|  MFLOPs(10^6 vector)         | 浮点计算次数(10^6 vector)|
|  GFLOPS(10^9 vector)         | 每秒浮点计算次数(10^9 vector)|

## output_timeline_data_*.txt文件说明

output_timeline_data_\*.txt文件包括device侧算子的耗时信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  kernel_name                 | device侧执行kernel算子全名|
|  stream_id                   | 算子所处Stream ID|
|  start_time                  | 算子执行开始时间(us)|
|  duration                    | 算子执行用时(ms)|

## cpu_ms_memory_record_*.txt文件说明

cpu_ms_memory_record_\*.txt文件包含应用级内存占用的信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  Timestamp                   | 内存事件发生时刻(ns)|
|  Total Allocated             | 内存分配总额(Byte)|
|  Total Reserved              | 内存预留总额(Byte)|
|  Total Active                | MindSpore中的流申请的总内存(Byte)|

## operator_memory_*.csv文件说明

operator_memory_\*.csv文件包含算子级内存占用的信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  Name                        | 内存占用Tensor名|
|  Size                        | 占用内存大小(KB)|
|  Allocation Time             | Tensor内存分配时间(us)|
|  Duration                    | Tensor内存占用时间(us)|
|  Allocation Total Allocated  | 算子内存分配时的内存分配总额(MB)|
|  Allocation Total Reserved   | 算子内存分配时的内存占用总额(MB)|
|  Release Total Allocated     | 算子内存释放时的内存分配总额(MB)|
|  Release Total Reserved      | 算子内存释放时的内存占用总额(MB)|
|  Device                      | device类型|