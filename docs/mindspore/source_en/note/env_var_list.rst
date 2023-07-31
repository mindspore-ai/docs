Environment Variables
=====================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png 
   :target: https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/env_var_list.rst

MindSpore environment variables are as follows:

Operators Compile
-----------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_BUILD_PROCESS_NUM
     - Specifies the number of parallel operator build processes during Ascend backend compilation.
       
       Note: Only Ascend backend.
     - Integer
     - The number of parallel operator build processes ranges from 1 to 24.
     - 
   * - MS_COMPILER_CACHE_ENABLE
     - Specifies whether to save or load the cache of the graph compiled by front-end. 
       The function is the same as the `enable_compile_cache <https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.set_context.html#mindspore.set_context>`_ in MindSpore context.

       Note: This environment variable has lower precedence than the context `enable_compile_cache`.
     - Integer
     - 0: Disable the compile cache

       1: Enable the compile cache
     - If it is used together with `MS_COMPILER_CACHE_PATH`, the directory for storing the cache files is `${MS_COMPILER_CACHE_PATH}` `/rank_${RANK_ID}` `/graph_cache/`. 
       `RANK_ID` is the unique ID for multi-cards training, the single card scenario defaults to `RANK_ID=0`.
   * - MS_COMPILER_CACHE_PATH
     - MindSpore compile cache directory and save the graph or operator cache files like `graph_cache`, `kernel_meta`, `somas_meta`.
     - String
     - File path, which can be a relative path or an absolute path.
     - 
   * - MS_COMPILER_OP_LEVEL
     - Enable debug function and generate the TBE instruction mapping file during Ascend backend compilation.

       Note: Only Ascend backend.
     - Integer
     - The value of compiler op level should be one of [0, 1, 2, 3, 4].

       0: Turn off op debug and delete op compile cache files

       1: Turn on debug, generate the `*.cce` and `*_loc.json`

       2: Turn on debug, generate the `*.cce` and `*_loc.json` files and turn off the compile optimization switch (The CCEC compiler option is set to `-O0-g`) at the same time

       3: Turn off op debug (default)

       4: Turn off op debug, generate the `*.cce` and `*_loc.json` files, generate UB fusion calculation description files (`{$kernel_name}_compute.json`) for fusion ops
     - When an AICore Error occurs, if you need to save the cce file of ops, you can set the `MS_COMPILER_OP_LEVEL` to 1 or 2
   * - MS_DEV_DISABLE_PREBUILD
     - Turn off operator prebuild processes during Ascend backend compilation. The prebuild processing may fix the attr `fusion_type` of the operate, and then affect the operator fusion. 
       If the performance of fusion operator can not meet the expectations, try to turn on this environment variable to verify if there is the performance problem of fusion operator.

       Note: Only Ascend backend.
     - Boolean
     - true: turn off prebuild

       false: enable prebuild
     - 

For more information, see `Incremental Operator Build <https://mindspore.cn/tutorials/experts/en/r2.1/optimize/op_compilation.html>`_ and `FAQ <https://mindspore.cn/docs/en/r2.1/faq/operators_compile.html>`_.

Parallel Training
-----------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - RANK_ID
     - Specifies the logical ID of the Ascend AI Processor called during deep learning.
     - Integer
     - The value ranges from 0 to 7. When multiple servers are running concurrently, `DEVICE_ID`s in different servers may be the same. 
       RANK_ID can be used to avoid this problem. `RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID`
     - 
   * - RANK_SIZE
     - Specifies the number of Ascend AI Processors to be called during deep learning.

       Note: When the Ascend AI Processor is used, specified by user when a distributed case is executed.
     - Integer
     - The number of Ascend AI Processors to be called ranges from 1 to 8.
     - This variable is used together with RANK_TABLE_FILE
   * - RANK_TABLE_FILE
     - Specifies the file to which a path points, including `device_ip` corresponding to multiple Ascend AI Processor `device_id`.

       Note: When the Ascend AI Processor is used, specified by user when a distributed case is executed.
     - String
     - File path, which can be a relative path or an absolute path.
     - This variable is used together with RANK_SIZE.
   * - MS_COMM_COMPILER_OPT
     - Specifies the maximum number of communication operators that can be replaced by corresponding communication subgraph during Ascend backend compilation in graph mode.

       Note: When the Ascend AI Processor is used, specified by user when a distributed case is executed.
     - Integer
     - -1 or an positive integer: communication subgraph extraction and reuse is enabled. -1 means that default value will be used. A positive integer means that the user specified value will be used.

       if not set: communication subgraph extraction and reuse is turned off.
     - 

For more information, see `Distributed Parallel Training Example <https://mindspore.cn/tutorials/experts/en/r2.1/parallel/train_ascend.html#running-the-script>`_.

Running Data Recorder
---------------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_RDR_ENABLE
     - Determines whether to enable running data recorder (RDR). 
       If a running exception occurs in MindSpore, the pre-recorded data in MindSpore is automatically exported to assist in locating the cause of the running exception.
     - Integer
     - 1：enables RDR
       
       0：disables RDR
     - This variable is used together with `MS_RDR_MODE` and `MS_RDR_PATH`.
   * - MS_RDR_MODE
     - Determines the exporting mode of running data recorder (RDR).
     - Integer
     - 1：export data when training process terminates in exceptional scenario

       2：export data when training process terminates in both exceptional scenario and normal scenario.
       
       Default: 1.
     - This variable is used together with `MS_RDR_ENABLE=1`.
   * - MS_RDR_PATH
     - Specifies the system path for storing the data recorded by running data recorder (RDR).
     - String
     - Directory path, which should be an absolute path.
     - This variable is used together with `MS_RDR_ENABLE=1`. The final directory for recording data is `${MS_RDR_PATH}` `/rank_${RANK_ID}/rdr/`. 
       `RANK_ID` is the unique ID for multi-cards training, the single card scenario defaults to `RANK_ID=0`.

For more information, see `Running Data Recorder <https://www.mindspore.cn/tutorials/experts/en/r2.1/debug/rdr.html>`_.

Log
---

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - GLOG_log_dir
     - Specifies the log level.
     - String
     - File path, which can be a relative path or an absolute path.
     - This variable is used together with GLOG_logtostderr

       If the value of `GLOG_logtostderr` is 0, this variable must be set
	   
       If `GLOG_log_dir` is specified and the value of `GLOG_logtostderr` is 1, the logs are output to the screen and not to the file
	 
       The log saving path is: `specified path/rank_${rank_id}/logs/`. Under non-distributed training scenario, `rank_id` is 0, while under distributed training scenario, `rank_id` is the ID of the current device in the cluster 

       C++ and Python logs are output to different files. The C++ logs follow the `GLOG` log file naming rules. In this case `mindspore.machine name. user name.log.log level.timestamp.Process ID`, the Python log file name is `mindspore.log.process ID`.
	   
       `GLOG_log_dir` can only contain upper and lower case letters, numbers, "-", "_", "/" characters, etc.
   * - GLOG_log_max
     - Control the size of the MindSpore C++ module log file. You can change the default maximum value of the log file with this environment variable
     - Integer
     - Positive integer. Default value: 50MB
     - If the current written log file exceeds the maximum value, the new output log content is written to a new log file
   * - GLOG_logtostderr
     - Specifies the log output mode.
     - Integer
     - 1: logs are output to the screen
       
       0: logs are output to a file

       Default: 1
     - This variable is used together with GLOG_log_dir
   * - GLOG_stderrthreshold
     - The log module will print logs to the screen when these logs are output to a file. This environment variable is used to control the log level printed to the screen in this scenario.
     - Integer
     - 0-DEBUG
       
       1-INFO

       2-WARNING

       3-ERROR

       4-CRITICAL

       Default: 2
     - 
   * - GLOG_v
     - Specifies the log level.
     - Integer
     - 0-DEBUG
       
       1-INFO

       2-WARNING

       3-ERROR, indicating that the program execution error, output error log, and the program may not terminate
	   
       4-CRITICAL, indicating that the execution of the program is abnormal, and the program may not terminate

       Default: 2.
     - After a log level is specified, output log messages greater than or equal to that level
   * - logger_backupCount
     - Controls the number of mindspire Python module log files.
     - Integer
     - Default: 30
     - 
   * - logger_maxBytes
     - Controls the size of the mindspire Python module log file.
     - Integer
     - Default: 52428800 bytes
     - 
   * - MS_SUBMODULE_LOG_v
     - Specifies log levels of C++ sub modules of MindSpore.
     - Dict {String:Integer...}
     - 0-DEBUG
       
       1-INFO

       2-WARNING

       3-ERROR
       
       SubModule: COMMON, MD, DEBUG, DEVICE, COMMON, IR...
     - The assignment way is:`MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`
	 
       The log level of the specified sub-module will override the setting of `GLOG_v` in this module, where the log level of the sub-module `LogLevel` has the same meaning as that of `GLOG_v`, and the division of MindSpore sub-module is as follows
	   
       For example, you can set the log level of `PARSER` and `ANALYZER` modules to WARNING and the log level of other modules to INFO by `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`.

The MindSpore submodules are divided by directory as follows:

.. list-table::
   :widths: 30 10
   :header-rows: 1

   * - Source Files
     - Sub Module Name
   * - mindspore/ccsrc/kernel
     - KERNEL
   * - mindspore/ccsrc/plugin/device/\*/kernel
     - KERNEL
   * - mindspore/ccsrc/backend/common/optimizer
     - PRE_ACT
   * - mindspore/ccsrc/backend/common/pass
     - PRE_ACT
   * - mindspore/ccsrc/plugin/device/\*/optimizer
     - PRE_ACT
   * - mindspore/ccsrc/backend/common/session
     - SESSION
   * - mindspore/ccsrc/common
     - COMMON
   * - mindspore/ccsrc/debug
     - DEBUG
   * - mindspore/ccsrc/frontend/operator
     - ANALYZER
   * - mindspore/ccsrc/frontend/optimizer
     - OPTIMIZER
   * - mindspore/ccsrc/frontend/parallel
     - PARALLEL
   * - mindspore/ccsrc/minddata/dataset
     - MD
   * - mindspore/ccsrc/minddata/mindrecord
     - MD
   * - mindspore/ccsrc/pipeline/jit/\*.cc
     - PIPELINE
   * - mindspore/ccsrc/pipeline/jit/parse
     - PARSER
   * - mindspore/ccsrc/pipeline/jit/static_analysis
     - ANALYZER
   * - mindspore/ccsrc/pipeline/pynative
     - PYNATIVE
   * - mindspore/ccsrc/pybind_api
     - COMMON
   * - mindspore/ccsrc/runtime/device
     - DEVICE
   * - mindspore/ccsrc/runtime/hardware
     - DEVICE
   * - mindspore/ccsrc/runtime/collective
     - DEVICE
   * - mindspore/ccsrc/runtime/pynative
     - DEVICE
   * - mindspore/ccsrc/runtime/addons
     - RUNTIME_FRAMEWORK
   * - mindspore/ccsrc/runtime/graph_scheduler
     - RUNTIME_FRAMEWORK
   * - mindspore/ccsrc/transform/graph_ir
     - GE_ADPT
   * - mindspore/ccsrc/transform/express_ir
     - EXPRESS
   * - mindspore/ccsrc/utils
     - UTILS
   * - mindspore/ccsrc/backend/graph_compiler
     - VM
   * - mindspore/ccsrc
     - ME
   * - mindspore/core
     - CORE

Note: glog does not support log file wrapping. If you need to control the log file occupation of disk space, you can use the log file management tool provided by the operating system, for example: logrotate for Linux.

Dump Function
-------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MINDSPORE_DUMP_CONFIG
     - Specify the path of the configuration file that the `cloud-side Dump <https://www.mindspore.cn/tutorials/experts/en/r2.1/debug/dump.html#synchronous-dump>`_
       or the `device-side Dump <https://www.mindspore.cn/lite/docs/en/r2.1/use/benchmark_tool.html#dump>`_ depends on.
     - String
     - File path, which can be a relative path or an absolute path.
     - 
   * - MS_DIAGNOSTIC_DATA_PATH
     - When the `cloud-side Dump <https://www.mindspore.cn/tutorials/experts/en/r2.1/debug/dump.html#synchronous-dump>`_ is enabled, 
       if the `path` field is not set or set to an empty string in the Dump configuration file, then `$MS_DIAGNOSTIC_DATA_PATH` `/debug_dump is regarded as path. 
       If the `path` field in configuration file is not empty, it is still used as the path to save Dump data.
     - String
     - File path, only absolute path is supported.
     - This variable is used together with MINDSPORE_DUMP_CONFIG.

For more information, see `Using Dump in the Graph Mode <https://www.mindspore.cn/tutorials/experts/en/r2.1/debug/dump.html>`_.

Data Processing
---------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - DATASET_ENABLE_NUMA
     - Determines whether to enable numa bind feature. Most of time this configuration can improve performance on distribute scenario.
     - String
     - True: Enables the numa bind feature.
     - This variable is used together with libnuma.so.
   * - MS_CACHE_HOST
     - Specifies the IP address of the host where the cache server is located when the cache function is enabled.
     - String
     - IP address of the host where the cache server is located.
     - This variable is used together with MS_CACHE_PORT.
   * - MS_CACHE_PORT
     - Specifies the port number of the host where the cache server is located when the cache function is enabled.
     - String
     - Port number of the host where the cache server is located.
     - This variable is used together with MS_CACHE_HOST.
   * - OPTIMIZE
     - Determines whether to optimize the pipeline tree for dataset during data processing. This variable can improve the data processing efficiency in the data processing operator fusion scenario.
     - String
     - true: enables pipeline tree optimization.

       false: disables pipeline tree optimization.
     - 

For more information, see `Single-Node Data Cache <https://mindspore.cn/tutorials/experts/en/r2.1/dataset/cache.html>`_ and `Optimizing the Data Processing <https://mindspore.cn/tutorials/experts/en/r2.1/dataset/optimize.html>`_.

Debugger
--------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - ENABLE_MS_DEBUGGER
     - Determines whether to enable Debugger during training.
     - Boolean
     - 1: enables Debugger.

       0: disables Debugger.
     - This variable is used together with MS_DEBUGGER_HOST and MS_DEBUGGER_PORT.
   * - MS_DEBUGGER_HOST
     - Specifies the IP of the MindSpore Insight Debugger Server.
     - String
     - IP address of the host where the MindSpore Insight Debugger Server is located.
     - This variable is used together with ENABLE_MS_DEBUGGER=1 and MS_DEBUGGER_PORT.
   * - MS_DEBUGGER_PARTIAL_MEM
     - Determines whether to enable partial memory overcommitment. (Memory overcommitment is disabled only for nodes selected on Debugger.)
     - Boolean
     - 1: enables memory overcommitment for nodes selected on Debugger.

       0: disables memory overcommitment for nodes selected on Debugger.
     - 
   * - MS_DEBUGGER_PORT
     - Specifies the port for connecting to the MindSpore Insight Debugger Server.
     - Integer
     - Port number ranges from 1 to 65536.
     - This variable is used together with ENABLE_MS_DEBUGGER=1 and MS_DEBUGGER_HOST.

For more information, see `Debugger <https://www.mindspore.cn/mindinsight/docs/en/r2.1/debugger.html>`_.

Other
-----

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - GROUP_INFO_FILE
     - Specify communication group information storage path
     - String
     - Communication group information file path, supporting relative path and absolute path.
     - 
   * - GRAPH_OP_RUN
     - When running the pipeline large network model in task sinking mode in graph mode, it may not be able to start as expected due to the limitation of stream resources. 
       This environment variable can specify the execution mode of the graph mode. 
       Set this variable to 0, indicating that model will be executed in non-task sinking mode which is the default execution mode. 
       Set this variable to 1, indicating a non-task sinking mode, which has no flow restrictions, but has degraded performance.
     - Integer
     - 0: task sinking mode.

       1: non-task sinking mode.
     - 
   * - MS_DEV_ENABLE_FALLBACK
     - Fallback function is enabled when the environment variable is set to a value other than 0.
     - Integer
     - 1: enables fallback function

       0: disables fallback function

       Default: 1
     - 
   * - MS_JIT_MODULES
     - Specifies which modules in static graph mode require JIT static compilation, and their functions and methods will be compiled into static calculation graphs.
     - String
     - The module name, corresponding to the name of the imported top-level module. If there are more than one, separate them with commas. For example, `export MS_JIT_MODULES=mindflow,mindyolo`.
     - By default, modules other than third-party libraries will be perform JIT static compilation, and MindSpore suites such as `mindflow` and `mindyolo` will not be treated as third-party libraries. If there is a module similar to MindSpore suites, which contains `nn.Cell`, `@ms.jit` decorated functions or functions to be compiled into static calculation graphs, you can configure the environment variable, so that the module will be perform JIT static compilation instead of being treated as third-party library.
   * - MS_JIT_IGNORE_MODULES
     - Specifies which modules are treated as third-party libraries in static graph mode without JIT static compilation. Their functions and methods will be interpreted and executed.
     - String
     - The module name, corresponding to the name of the imported top-level module. If there are more than one, separate them with commas. For example, `export MS_JIT_IGNORE_MODULES=numpy,scipy`.
     - Static graph mode can automatically recognize third-party libraries, and generally there is no need to set this environment variable for recognizable third-party libraries such as NumPy and Scipy. If `MS_JIT_IGNORE_MODULES` and `MS_JIT_MODULES` specify the same module name at the same time, the former takes effect and the latter does not.
   * - MS_EXCEPTION_DISPLAY_LEVEL
     - Control the display level of exception information
     - Integer
     - 0: display exception information related to model developers and framework developers

       1: display exception information related to model developers

       Default: 0
     - 
   * - MS_OM_PATH
     - Specifies the save path for the file `analyze_fail.ir/*.npy` which is dumped if task exception or a compiling graph error occurred. 
       The file will be saved to the path of `the_specified_directory` `/rank_${rank_id}/om/`.
     - String
     - File path, which can be a relative path or an absolute path.
     - 
   * - OPTION_PROTO_LIB_PATH
     - Specifies the RPOTO dependent library path.
     - String
     - File path, which can be a relative path or an absolute path.
     - 
   * - MS_KERNEL_LAUNCH_SKIP
     - Specifies the kernel or subgraph to skip during execution.
     - String
     - ALL or all: skip the execution of all kernels and subgraphs

       kernel name (such as ReLU) : skip the execution of all ReLU kernels

       subgraph name (such as kernel_graph_1) : skip the execution of subgraph kernel_graph_1, used for subgraph sink mode
     - 
   * - MS_DEV_SAVE_GRAPTHS_SORT_MODE
     - Choose the sort mode of the graphs printed in the ir files.
     - Integer
     - 0: print default ir file

       1: print deep sorted ir file
     - 