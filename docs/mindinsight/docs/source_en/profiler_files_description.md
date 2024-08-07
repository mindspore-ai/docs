# Performance Data in the Profiler Directory

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_en/profiler_files_description.md)

The profiler directory includes some csv, json, and txt files, which contain performance data such as operator execution time, memory usage, and communication during model computation to help users analyze performance bottlenecks. The fields in some of the csv and txt files are explained below. The contents of the files mainly include information about the time consumed by the device-side operators (AI Core operator and AI CPU operator), and the information about the memory occupied at the operator level and the memory occupied at the application level.

## aicore_intermediate_*_detail.csv File Descriptions

The aicore_intermediate_\*_detail.csv file contains statistical AI Core operator information based on the contents of output_timeline_data_\*.txt and framework_raw_\*.csv. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|----------------------------|----------------------------|
|full_kernel_name            |Full name of the device-side execution kernel operator|
|task_duration               |Operator execution time|
|execution_frequency         |Frequency of operator execution|
|task_type                   |Task types for the operator|

## aicore_intermediate_*_type.csv File Descriptions

The aicore_intermediate_\*_type.csv file includes statistical information about the specific types of AI Core operators based on the contents of output_timeline_data_\*.txt and framework_raw_\*.csv. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  kernel_type                 | AI Core operator types|
|  task_time                   | Operator execution time|
|  execution_frequency         | Frequency of operator execution|
|  percent                     | Percentage of time taken of this operator type over the total time taken of all operators|

## aicpu_intermediate_*.csv File Descriptions

The aicpu_intermediate_\*.csv file contains time taken information of the AI CPU operators. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  serial_num                  | AI CPU operator number|
|  kernel_type                 | AI CPU operator type|
|  total_time                  | Operator time taken, which is equal to the sum of the downstream time taken and the execution time taken|
|  dispatch_time               | downstream time taken|
|  execution_time              |  execution time taken|
|  run_start                   | Start time of operator execution|
|  run_end                     | End time of operator execution|

## flops_*.txt File Descriptions

The flops_\*.txt file contains information about the number of floating-point computations for the device-side operator, the number of floating-point computations per second, and so on. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  full_kernel_name            | Full name of the device-side execution kernel operator|
|  MFLOPs(10^6 cube)           | The number of floating-point calculations (10^6 cube)|
|  GFLOPS(10^9 cube)           | The number of floating-point calculations per second(10^9 cube)|
|  MFLOPs(10^6 vector)         | The number of floating-point calculations(10^6 vector)|
|  GFLOPS(10^9 vector)         | The number of floating-point calculations per second(10^9 vector)|

## output_timeline_data_*.txt File Descriptions

The output_timeline_data_\*.txt file contains information about the time taken of the device-side operator. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  kernel_name                 | Full name of the device-side execution kernel operator|
|  stream_id                   | Stream ID of the operator|
|  start_time                  | Start time of operator execution(us)|
|  duration                    | operator execution time (ms)|

## cpu_ms_memory_record_*.txt File Descriptions

The cpu_ms_memory_record_\*.txt file contains information about application-level memory usage. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  Timestamp                   | Moment of memory event(ns)|
|  Total Allocated             | Total memory allocation(Byte)|
|  Total Reserved              | Total memory reservation(Byte)|
|  Total Active                | Total memory requested by streams in MindSpore(Byte)|

## operator_memory_*.csv File Descriptions

The operator_memory_\*.csv file contains information about operator-level memory usage. Refer to the following table for descriptions of the fields in the file:

| Field Names                     | Descriptions |
|------------------------------|------------------------|
|  Name                        | Memory Consumption Tensor Name|
|  Size                        | Size of memory occupied(KB)|
|  Allocation Time             | Tensor memory allocation time(us)|
|  Duration                    | Tensor memory occupation time(us)|
|  Allocation Total Allocated  | Total memory allocation at operator memory allocation(MB)|
|  Allocation Total Reserved   | Total memory occupation at operator memory allocation(MB)|
|  Release Total Allocated     | Total memory allocation at operator memory release(MB)|
|  Release Total Reserved      | Total memory occupation at operator memory release(MB)|
|  Device                      | device type|
