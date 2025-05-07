Environment Variables
=====================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/api_python/env_var_list.rst
    :alt: View Source On Gitee

MindSpore environment variables are as follows:

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
     - Determines whether to enable numa feature for dataset module. Most of time this configuration can improve performance on distribute scenario.
     - String
     - True: Enables the numa feature for dataset module.
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
   * - MS_DATASET_SINK_QUEUE
     - Specifies the size of data queue in sink mode.
     - Integer
     - 1~128: Valid range of queue size.
     -
   * - MS_DEV_MINDRECORD_SHARD_BY_BLOCK
     - When performing sharding sampling on MindRecord, whether to switch the slicing strategy of the data to sampling by block.
     - String
     - True: The data segmentation strategy is block sampling.

       False: The data segmentation strategy is slice sampling.
     - Default: False. This only applies to the DistributedSampler sampler.
   * - MS_ENABLE_NUMA
     - Whether to enable numa feature in global context to improve end-to-end performance.
     - String
     - True: Enables the numa feature in global context.
     -
   * - MS_FREE_DISK_CHECK
     - Whether to enable the check of disk space.
     - String
     - True: Enable the check of disk space.

       False: Disable the check of disk space.
     - Default: True, When creating MindRecords on shared storage using multiple concurrent operations, it is recommended to set it to False.
   * - MS_INDEPENDENT_DATASET
     - Whether to enable dataset independent process mode. Dataset will run in independent child processes. Only supports Linux platform.
     - String
     - True: Enable the dataset independent process mode.

       False: Disable the dataset independent process mode.
     - Default: False. This feature is currently in beta testing. Does not support use with AutoTune, Offload, Cache or DSCallback. If you encounter any problems during use, please feel free to provide feedback.
   * - OPTIMIZE
     - Determines whether to optimize the pipeline tree for dataset during data processing. This variable can improve the data processing efficiency in the data processing operator fusion scenario.
     - String
     - true: enables pipeline tree optimization.

       false: disables pipeline tree optimization.
     -

For more information, see `Single-Node Data Cache <https://mindspore.cn/tutorials/en/master/dataset/cache.html>`_ and `Optimizing the Data Processing <https://mindspore.cn/tutorials/en/master/dataset/optimize.html>`_.

Graph Compilation and Execution
---------------------------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_DEV_JIT_SYNTAX_LEVEL
     - Specify the syntax support level of static graph mode.
     - Integer
     - 0: Specify the syntax support level of static graph mode as STRICT level. Only basic syntaxes is supported, and execution performance is optimal. Can be used for MindIR load and export.

       2: Specify the syntax support level of static graph mode as LAX level. More complex syntaxes are supported, compatible with all Python syntax as much as possible. Cannot be used for MindIR load and export due to some syntax that may not be able to be exported.
     -
   * - MS_JIT_MODULES
     - Specify which modules in static graph mode require JIT static compilation, and their functions and methods will be compiled into static calculation graphs.
     - String
     - The module name, corresponding to the name of the imported top-level module. If there are more than one, separate them with commas. For example, `export MS_JIT_MODULES=mindflow,mindyolo`.
     - By default, modules other than third-party libraries will be perform JIT static compilation, and MindSpore suites such as `mindflow` and `mindyolo` will not be treated as third-party libraries. If there is a module similar to MindSpore suites, which contains `nn.Cell`, `@ms.jit` decorated functions or functions to be compiled into static calculation graphs, you can configure the environment variable, so that the module will be perform JIT static compilation instead of being treated as third-party library.
   * - MS_JIT_IGNORE_MODULES
     - Specify which modules are treated as third-party libraries in static graph mode without JIT static compilation. Their functions and methods will be interpreted and executed.
     - String
     - The module name, corresponding to the name of the imported top-level module. If there are more than one, separate them with commas. For example, `export MS_JIT_IGNORE_MODULES=numpy,scipy`.
     - Static graph mode can automatically recognize third-party libraries, and generally there is no need to set this environment variable for recognizable third-party libraries such as NumPy and Scipy. If `MS_JIT_IGNORE_MODULES` and `MS_JIT_MODULES` specify the same module name at the same time, the former takes effect and the latter does not.
   * - MS_DEV_FALLBACK_DUMP_NODE
     - Print syntax expressions supported by Static Graph Syntax Enhancement in the code.
     - Integer
     - 1: Enable printing.

       No setting or other value: Disable printing.
     -
   * - MS_JIT
     - Specify whether to use just-in-time compilation.
     - Integer
     - 0: Do not use just-in-time compilation, and the network script is executed directly in dynamic graph (PyNative) mode.

       No setting or other value: Determine whether to execute static graph (Graph) mode or dynamic graph (PyNative) mode according to the network script.
     -
   * - MS_DEV_FORCE_USE_COMPILE_CACHE
     - Specify whether to use the compilation cache directly without checking whether the network script has been modified.
     - Integer
     - 1: Do not check whether the network script has been modified, directly use the compilation cache. It is recommended to only use it during debugging. For example, the network script only adds print statements for printing and debugging.

       No setting or other value: Detect changes in network scripts, and only use the compilation cache when the network scripts have not been modified.
     -
   * - MS_DEV_SIDE_EFFECT_LOAD_ELIM
     - Optimize redundant memory copy operations.
     - Integer
     - 0: Do not do video memory optimization, occupy the most video memory.

       1: Conservatively do some memory optimization.

       2: Under the premise of losing a certain amount of compilation performance, optimize the video memory as much as possible.

       3: The accuracy of the network is not guaranteed, and the memory consumption is minimal.

       Default: 1
     -
   * - MS_DEV_SAVE_GRAPHS
     - Specify whether to save IR files.
     - Integer
     - 0: Disable saving IR files.

       1: Some intermediate files will be generated during graph compilation.

       2: Based on level1, generate more IR files related to backend process.

       3: Based on level2, generate visualization computing graphs and detailed frontend IR graphs.
     -
   * - MS_DEV_SAVE_GRAPHS_PATH
     - Specify path to save IR files.
     - String
     - Path to save IR files.
     -
   * - MS_DEV_DUMP_IR_FORMAT
     - Configure what information is displayed in IR graphs.
     - Integer
     - 0: Except for the return node, only the operator and inputs of the node are displayed, and the detailed information of subgraph is simplified.

       1: Display all information except debug info and scope.

       2 or not set: Display all information.
     -
   * - MS_DEV_DUMP_IR_INTERVAL
     - Set to save an IR file every few IR files to reduce the number of IR files.
     - Integer
     - 1 or not set: Save all IR files.

       Other values: Save IR files at specified intervals.
     - When this environment variable is enabled together with MS_DEV_DUMP_IR_PASSES, the rules of MS_DEV_DUMP_IR_PASSES take priority, and this environment variable will not take effect.
   * - MS_DEV_DUMP_IR_PASSES
     - Specify which IR files to save based on the file name.
     - String
     - Pass's name of part of its name. If there are multiple, use commas to separate them. For example, `export MS_DEV_DUMP_IR_PASSES=recompute,renormalize`.
     - When setting this environment variable, regardless of the value of MS_DEV_SAVE_GRAPHS, detailed frontend IR files will be filtered and printed.
   * - MS_DEV_DUMP_IR_PARALLEL_DETAIL
     - Set to print detailed information about the DUMP IR, image tensor_map and device_matrix.

     - Integer
     - 1: Print detailed information about the DUMP IR, including inputs_tensor_map, outputs_tensor_map, and device_matrix.

       No setting or other value: Not print detailed information of Dump IR.
     -
   * - MS_JIT_DISPLAY_PROGRESS
     - Specify whether to print compilation progress information.
     - Integer
     - 1: Print main compilation progress information.

       No setting or other value: Do not print compilation progress information.
     -
   * - MS_DEV_PRECOMPILE_ONLY
     - Specify whether the network is precompiled only and not executed.
     - Integer
     - 1: The network is precompiled only and not executed.

       No setting or other value: Do not precompile the network, that is, compile and execute the network.
     -
   * - MS_KERNEL_LAUNCH_SKIP
     - Specifies the kernel or subgraph to skip during execution.
     - String
     - ALL or all: skip the execution of all kernels and subgraphs

       kernel name (such as ReLU) : skip the execution of all ReLU kernels

       subgraph name (such as kernel_graph_1) : skip the execution of subgraph kernel_graph_1, used for subgraph sink mode
     -
   * - GC_COLLECT_IN_CELL
     - Whether to perform garbage collection on unused Cell objects
     - Integer
     - 1: Perform garbage collection on unused Cell objects

       No setting or other value: not calling the garbage collection
     - This environment variable will be removed subsequently and is not recommended.
   * - MS_DEV_USE_PY_BPROP
     - The op which set by environment will use python bprop instead of cpp expander bprop
     - String
     - Op name, can set more than one name, split by ','
     - Experimental environment variable. It will run fail when python bprop does not exist
   * - MS_DEV_DISABLE_BPROP_CACHE
     - Disable to use bprop's graph cache
     - String
     - 'on', indicating that disable to use bprop's graph cache
     - Experimental environment variable. When set env on, it will slow down building bprop's graph
   * - MS_ENABLE_IO_REUSE
     - Turn on the graph input/output memory multiplexing flag
     - Integer
     - 1: Enable this function.

       0: not enabled.

       Default value: 0
     - Ascend AI processor environment and graph compilation grade O2 process use only.
   * - MS_ENABLE_GRACEFUL_EXIT
     - Enable training process exit gracefully
     - Integer
     - 1: Enable graceful exit.

       No setting or other value: Disable graceful exit.
     - Rely on the callback function to enable graceful exit. Refer to the `Example of Graceful Exit <https://www.mindspore.cn/tutorials/en/master/train_availability/graceful_exit.html>`_ .
   * - MS_DEV_BOOST_INFER
     - Compile optimization switch for graph compilation. This switch accelerates the type inference module to speed up network compilation.
     - Integer
     - 0: Disables the optimization.

       No setting or other value: Enables the optimization.
     - This environment variable will be removed subsequently.

   * - MS_DEV_RUNTIME_CONF
     - Configure the runtime environment.
     - String
     - Configuration items, with the format "key: value", multiple configuration items separated by commas, for example, "export MS_DEV_RUNTIME_CONF=inline:false,pipeline:false".

       inline: In the scenario of sub image cell sharing, whether to enable backend inline, only effective in O0 or O1 mode, with a default value of true.

       switch_inline: Whether to enable backend control flow inline, only effective in O0 or O1 mode, with a default value of true.

       multi_stream: The backend stream diversion method, with possible values being 1) true: One stream for communication and one for computation. 2) false: Disable multi-streaming, use a single stream for both communication and computation. 3) group (default value): Communication operators are diverted based on their communication domain.

       pipeline: Whether to enable runtime pipeline, only effective in O0 or O1 mode, with a default value of true.

       all_finite: Whether to enable Allfitine in overflow detection, only effective in O0 or O1 mode, with a default value of true.

       memory_statistics: Whether to enable memory statistics, with a default value of false.

       compile_statistics: Whether to enable compile statistics, with a default value of false.

       backend_compile_cache: Whether to enable backend cache in O0/O1 mode, only effective when enable complie cache(MS_COMPILER_CACHE_ENABLE), with a default value of true.

       view: Whether to enable view kernels, only effective in O0 or O1 mode, with a default value of true.
     -
   * - MS_DEV_VIEW_OP
     - Specify certain operators to replace by view with MS_DEV_RUNTIME_CONF enabled view
     - String
     - Op name, can set more than one name, split by ','
     - Experimental environment variable.

   * - MS_ALLOC_CONF
     - Configure the memory allocation.
     - String
     - Configuration items, with the format "key: value", multiple configuration items separated by commas, for example, "export MS_ALLOC_CONF=enable_vmm:true,memory_tracker:true".

       enable_vmm: Whether to enable virtual memory, with a default value of true.

       vmm_align_size: Set the virtual memory alignment size in MB, with a default value of 2.

       memory_tracker: Whether to enable memory tracker, with a default value of false.

       acl_allocator: Whether to enable ACL memory allocator, with a default value of true.

       somas_whole_block: Whether to use the entire Somas for memory allocation, with a default value of false.
     -

   * - MS_DEV_GRAPH_KERNEL_FLAGS
     - Configure the graph kernel fusion strategy.
     - String
     - Configuration items, with the format "--key=value", multiple configuration items separated by space, multiple value items separated by commas, for example, `export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_expand_ops=Square --enable_cluster_ops=MatMul,Add"`

       opt_level: Set the optimization level. Default: `2` .

       enable_expand_ops: Forcefully expand operators that are not in the default list, requiring an expander implementation for the corresponding operator.

       disable_expand_ops: Disable the expansion of the specified operators.

       enable_expand_ops_only: Allow only the specified operators to expand. When this option is set, the above two options are ignored.

       enable_cluster_ops: Add specified operators to the set of operators participating in fusion based on the default fusion operator list.

       disable_cluster_ops: Prevent the specified operators from participating in the fusion set.

       enable_cluster_ops_only: Allow only the specified operators to participate in the fusion set. When this option is set, the above two options are ignored.

       disable_fusion_pattern: Prevent the specified fusion pattern from participating in the fusion set.

       enable_fusion_pattern_only: Allow only the specified fusion pattern to participate in the fusion set. When this option is set, the above option is ignored.

       enable_pass: Enable passes that are disabled by default using this option.

       disable_pass: Disable passes that are enabled by default using this option.

       dump_as_text: Save detailed information about key processes as text files in the `graph_kernel_dump` directory. Default value: `False`.

       enable_debug_mode: Insert synchronization points before and after the graph kernel mod launch, and print debugging information if the launch fails. This is supported only for the GPU backend. Default value: `False`.

       path: use specified json file. When this option is set, the above options are ignored.
     - Refer to the `Custom Fusion <https://www.mindspore.cn/tutorials/en/master/custom_program/fusion_pass.html>`_

Dump Debugging
---------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MINDSPORE_DUMP_CONFIG
     - Specify the path of the configuration file that the `cloud-side Dump <https://www.mindspore.cn/tutorials/en/master/debug/dump.html>`_
       or the `device-side Dump <https://www.mindspore.cn/lite/docs/en/master/tools/benchmark_tool.html#dump>`_ depends on.
     - String
     - File path, which can be a relative path or an absolute path.
     -
   * - MS_DIAGNOSTIC_DATA_PATH
     - When the `cloud-side Dump <https://www.mindspore.cn/tutorials/en/master/debug/dump.html>`_ is enabled,
       if the `path` field is not set or set to an empty string in the Dump configuration file, then `$MS_DIAGNOSTIC_DATA_PATH` `/debug_dump` is regarded as path.
       If the `path` field in configuration file is not empty, it is still used as the path to save Dump data.
     - String
     - File path, only absolute path is supported.
     - This variable is used together with MINDSPORE_DUMP_CONFIG.
   * - MS_DEV_DUMP_BPROP
     - Dump bprop ir file in current path
     - String
     - 'on', indicating that dump bprop ir file in current path
     - Experimental environment variable.
   * - ENABLE_MS_DEBUGGER
     - Determines whether to enable Debugger during training.
     - Boolean
     - 1: enables Debugger.

       0: disables Debugger.
     - This variable is used together with MS_DEBUGGER_HOST and MS_DEBUGGER_PORT.
   * - MS_DEBUGGER_PARTIAL_MEM
     - Determines whether to enable partial memory overcommitment. (Memory overcommitment is disabled only for nodes selected on Debugger.)
     - Boolean
     - 1: enables memory overcommitment for nodes selected on Debugger.

       0: disables memory overcommitment for nodes selected on Debugger.
     -
   * - MS_OM_PATH
     - Specifies the save path for the file `analyze_fail.ir/*.npy` which is dumped if task exception or a compiling graph error occurred.
       The file will be saved to the path of `the_specified_directory` `/rank_${rank_id}/om/`.
     - String
     - File path, which can be a relative path or an absolute path.
     -
   * - MS_DUMP_SLICE_SIZE
     - Specify slice size of operator Print, TensorDump, TensorSummary, ImageSummary, ScalarSummary, HistogramSummary.
     - Integer
     - 0~2048, unit: MB, default value is 0. The value 0 means the data is not sliced.
     -
   * - MS_DUMP_WAIT_TIME
     - Specify wait time of second stage for operator Print, TensorDump, TensorSummary, ImageSummary, ScalarSummary, HistogramSummary.
     - Integer
     - 0~600, unit: Seconds, default value is 0. The value 0 means using default wait time, i.e. the value of `mindspore.get_context("op_timeout")`.
     - This environment variable only takes effect when value of `MS_DUMP_SLICE_SIZE` is greater than 0. Now the wait time can not exceed value of `mindspore.get_context("op_timeout")`.

For more information, see `Using Dump in the Graph Mode <https://www.mindspore.cn/tutorials/en/master/debug/dump.html>`_.

Distributed Parallel
---------------------

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
     - The value ranges from 0 to 7. When multiple servers are running concurrently, `DEVICE_ID` in different servers may be the same.
       RANK_ID can be used to avoid this problem. `RANK_ID = SERVER_ID * DEVICE_NUM + DEVICE_ID`, and DEVICE_ID indicates the sequence number of the Ascend AI processor of the current host.
     -
   * - RANK_SIZE
     - Specifies the number of Ascend AI Processors to be called during deep learning.

       Note: When the Ascend AI Processor is used, specified by user when a distributed case is executed.
     - Integer
     - The number of Ascend AI Processors to be called ranges from 1 to 8.
     - This variable is used together with RANK_TABLE_FILE
   * - RANK_TABLE_FILE or MINDSPORE_HCCL_CONFIG_PATH
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

       Do not set or set other values:: communication subgraph extraction and reuse is turned off.
     -
   * - DEVICE_ID
     - The ID of the Ascend AI processor, which is the Device's serial number on the AI server.
     - Integer
     - The ID of the Rise AI processor, value range: [0, number of actual Devices-1].
     -
   * - MS_ROLE
     - Specifies the role of this process.
     - String
     - MS_SCHED: represents the Scheduler process, a training task starts only one Scheduler, which is responsible for networking, disaster recovery, etc., and does not execute the training code.

       MS_WORKER: represents the Worker process, which generally sets up the distributed training process for this role.

       MS_PSERVER: represents the Parameter Server process, and this role is only valid in Parameter Server mode.
     - The Worker and Parameter Server processes register with the Scheduler process to complete the networking.
   * - MS_SCHED_HOST
     - Specifies the IP address of the Scheduler.
     - String
     - Legal IP address.
     - The current version does not support IPv6 addresses.
   * - MS_SCHED_PORT
     - Specifies the Scheduler binding port number.
     - Integer
     - Port number in the range of 1024 to 65535.
     -
   * - MS_NODE_ID
     - Specifies the ID of this process, unique within the cluster.
     - String
     - Represents the unique ID of this process, which is automatically generated by MindSpore by default.
     - MS_NODE_ID needs to be set in the following cases. Normally it does not need to be set and is automatically generated by MindSpore:

       Enable Disaster Recovery Scenario: Disaster recovery requires obtaining the current process ID and thus re-registering with the Scheduler.

       Enable GLOG log redirection scenario: In order to ensure that the logs of each training process are saved independently, it is necessary to set the process ID, which is used as the log saving path suffix.

       Specify process rank id scenario: users can specify the rank id of this process by setting MS_NODE_ID to some integer.
   * - MS_WORKER_NUM
     - Specifies the number of processes with the role MS_WORKER.
     - Integer
     - Integers greater than 0.
     - The number of Worker processes started by the user should be equal to the value of this environment variable. If it is less than this value, the networking fails; if it is greater than this value, the Scheduler process will complete the networking according to the order of Worker registration, and the redundant Worker processes will fail to start.
   * - MS_SERVER_NUM
     - Specifies the number of processes with the role MS_PSERVER.
     - Integer
     - Integers greater than 0.
     - The setting is only required in Parameter Server training mode.
   * - MS_INTERFERED_SAPP
     - Turn on interfered sapp.
     - Integer
     - 1 for on. No setting or other value: off.
     -
   * - MS_ENABLE_RECOVERY
     - Turn on disaster tolerance.
     - Integer
     - 1 for on, 0 for off. The default is 0.
     -
   * - MS_RECOVERY_PATH
     - Persistent path folder.
     - String
     - Legal user directory.
     - The Worker and Scheduler processes perform the necessary persistence during execution, such as node information for restoring the grouping and training the intermediate state of the service, and are saved via files.
   * - GROUP_INFO_FILE
     - Specify communication group information storage path
     - String
     - Communication group information file path, supporting relative path and absolute path.
     -
   * - MS_SIMULATION_LEVEL
     - Specifies the simulation compilation level.
     - Integer
     - when set to 0, it will simulate graph compilation without occupying the card; when set to 1, it will simulate graph and kernel compilation without occupying the card; when set to 2, it will occupy the card and simulate graph and kernel compilation, making memory analysis more accurate; when set to 3, it will occupy the card and simulate the execution of kernels except communication kernels. Not enabled by default.
     - This environment variable is mainly used for single-card simulation of distributed multi-card specific rank card compilation scenarios and requires RANK_SIZE and RANK_ID to be used in conjunction with it.
   * - DUMP_PARALLEL_INFO
     - Enable dump parallel-related communication information in auto-parallel/semi-automatic parallelism mode. The dump path can be set by the environment variable `MS_DEV_SAVE_GRAPHS_PATH`.
     - Integer
     - 1: Enable dump parallel information.

       No setting or other value: Disable printing.
     - The JSON file saved by each card contains the following fields:

       hccl_algo: Ensemble communication algorithm.

       op_name: The name of the communication operator.

       op_type: The type of communication operator.

       shape: The shape information of the communication operator.

       data_type: The data type of the communication operator.

       global_rank_id: the global rank number.

       comm_group_name: the communication domain name of the communication operator.

       comm_group_rank_ids: The communication domain of the communication operator.

       src_rank: The rank_id of peer operator of the Receive operator.

       dest_rank: The rank_id of peer opposite of the Send operator.

       sr_tag: The identity ID of different send-receive pairs when src and dest are the same.
   * - MS_CUSTOM_DEPEND_CONFIG_PATH
     - Insert the control edge based on the configuration file xxx.json specified by the user, and use the primitive ops.Depend in MindSpore expresses the dependency control relationship.
     - String
     - This environment variable is only enabled in Atlas A2 series product graph mode.
     - The fields contained in the json file have the following meanings:

       get_full_op_name_list(bool): Whether to generate an operator name list, optional, default is false.

       stage_xxx(string): used in multi-card and multi-graph scenarios, that is, different cards execute different graphs (such as pipeline parallelism), where stage_xxx is just a serial number label, and the serial number value has no actual pointing meaning.

       graph_id (int): used to distinguish subgraph information. The graph_id number needs to be consistent with the actually executed graph_id. If it is inconsistent, the action of inserting control edges will be invalid.

       depend_src_list(List[string]): A list of source operator names that need to be inserted into control edges. They need to correspond one-to-one with the operators in depend_dest_list in order, otherwise the action of inserting control edges will fail.

       depend_dest_list(List[string]): A list of terminal operator names that need to be inserted into control edges. They need to correspond one-to-one with the operators in depend_src_list in order, otherwise the action of inserting control edges will fail.

       delete_depend_list(List[string]): A list of operator names that need to be deleted. If the operator name does not exist or does not match the graph_id, the action of deleting the node will be invalid.


See `Dynamic Cluster <https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html>`_ for more details about Dynamic Cluster.

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
   * - MS_COMPILER_CACHE_ENABLE
     - Whether to save or load the compiled cache of the graph. After `MS_COMPILER_CACHE_ENABLE` is set to `1`, during the first execution, a compilation cache
       is generated and exported to a MINDIR file. When the network is executed again, if `MS_COMPILER_CACHE_ENABLE` is still set to `1` and the network scripts
       are not changed, the compile cache is loaded.

       Note: Only limited automatic detection for the changes of python scripts is supported by now, which means that there is a correctness risk. Currently, do not
       support the graph which is larger than 2G after compiled. This is an experimental prototype that is subject to change and/or deletion.
     - Integer
     - 0: Disable the compile cache

       1: Enable the compile cache
     - If it is used together with `MS_COMPILER_CACHE_PATH`, the directory for storing the cache files is `${MS_COMPILER_CACHE_PATH}` `/rank_${RANK_ID}`.
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
   * - MS_ASCEND_CHECK_OVERFLOW_MODE
     - Setting the output mode of floating-point calculation results
     - String
     - SATURATION_MODE: Saturation mode.

       INFNAN_MODE: INF/NAN mode.

       Default value: INFNAN_MODE.

     - Saturation mode: Saturates to floating-point extremes (+-MAX) when computation overflows.

       INF/NAN mode: Follows the IEEE 754 standard and outputs INF/NAN calculations as defined.

       Atlas A2 training series use only.
   * - MS_CUSTOM_AOT_WHITE_LIST
     - Specify the valid path for custom operators to use dynamic libraries.
     - String
     - The path to validated dynamic libraries. The framework will validate based on the valid path specified for dynamic libraries used by custom operators. If the dynamic library used by a custom operator is not located in the specified path, the framework will report an error and refuse to use the corresponding dynamic library. When this setting is left empty, no validation will be performed on the dynamic libraries of custom operators.

       Default value: empty string.
     -

For more information, see `FAQ <https://mindspore.cn/docs/en/master/faq/operators_compile.html>`_.

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
   * - GLOG_max_log_size
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
   * - VLOG_v
     - Specifies the MindSpore verbose log level, configure the environment variable using export before `import mindspore`.
     - String
     - By command:
       `export VLOG_v=20000;python -c 'import mindspore';` view the available verbose log levels for MindSpore.
     - format1: `VLOG_v=number`: Only logs whose verbose level value is `number` will be output.

       format2: `VLOG_v=(number1,number2)`: Only logs whose verbose level is between `number1` and `number2` (including `number1` and `number2`) are output. Specially, `VLOG_v=(,number2)` outputs logs with verbose levels ranging from `1 to number2`, while `VLOG_v=(number1,)` outputs logs with verbose levels ranging from `number1 to 0x7fffffff`.

       The value of `number`, `number1` and `number2` must be a non-negative decimal integer. The maximum value is `0x7fffffff` the maximum value of the `int` type. Value of `VLOG_v` can not contain whitespace characters.

       Note: Braces `()` is special for bash, when exporting `VLOG_v` variable containing `()`, need use `'` or `"` to wrap it, for example, `export VLOG_v="(number1,number2)"` or `export VLOG_v='(number1,number2)'`. If put environment in the commandline, the quotation marks, `'` and `"`, are not necessary, for example, execute command `VLOG_v=(1,) python -c 'import mindspore'` to display the verbose tag already used by MindSpore.
   * - logger_backupCount
     - Controls the number of mindspore Python module log files.
     - Integer
     - Default: 30
     -
   * - logger_maxBytes
     - Controls the size of the mindspore Python module log file.
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

     - The assignment way is:`MS_SUBMODULE_LOG_v="{SubModule1:LogLevel1,SubModule2:LogLevel2,...}"`

       The log level of the specified sub-module will override the setting of `GLOG_v` in this module, where the log level of the sub-module `LogLevel` has the same meaning as that of `GLOG_v`. For a detailed list of MindSpore sub-modules, see `sub-module_names <https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/utils/log_adapter.cc>`_.

       For example, you can set the log level of `PARSER` and `ANALYZER` modules to WARNING and the log level of other modules to INFO by `GLOG_v=1 MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"`.
   * - GLOG_logfile_mode
     - The GLOG environment variable used to control the permissions of the GLOG log files in MindSpore
     - octal number
     - Refer to the numerical representation of the Linux file permission setting, default value: 0640 (value taken)
     -
   * - MS_RDR_ENABLE
     - Determines whether to enable running data recorder (RDR).
       If a running exception occurs in MindSpore, the pre-recorded data in MindSpore is automatically exported to assist in locating the cause of the running exception.
     - Integer
     - 1: enables RDR

       0: disables RDR
     - This variable is used together with `MS_RDR_MODE` and `MS_RDR_PATH`.
   * - MS_RDR_MODE
     - Determines the exporting mode of running data recorder (RDR).
     - Integer
     - 1: export data when training process terminates in exceptional scenario

       2: export data when training process terminates in both exceptional scenario and normal scenario.

       Default: 1.
     - This variable is used together with `MS_RDR_ENABLE=1`.
   * - MS_RDR_PATH
     - Specifies the system path for storing the data recorded by running data recorder (RDR).
     - String
     - Directory path, which should be an absolute path.
     - This variable is used together with `MS_RDR_ENABLE=1`. The final directory for recording data is `${MS_RDR_PATH}` `/rank_${RANK_ID}/rdr/`.
       `RANK_ID` is the unique ID for multi-cards training, the single card scenario defaults to `RANK_ID=0`.
   * - MS_EXCEPTION_DISPLAY_LEVEL
     - Control the display level of exception information
     - Integer
     - 0: display exception information related to model developers and framework developers

       1: display exception information related to model developers

       Default: 0
     -

Note: glog does not support log file wrapping. If you need to control the log file occupation of disk space, you can use the log file management tool provided by the operating system, for example: logrotate for Linux. Please set the log environment variables before `import mindspore` .

Feature Value Detection
------------------------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value
     - Description
   * - NPU_ASD_ENABLE
     - Whether to enable feature value detection function
     - Integer
     - 0: Disable feature value detection function

       1: Enable feature value detection function, when error was detected, just print log, not thow exception

       2: Enable feature value detection function, when error was detected, thow exception

       3: Enable feature value detection function, when error was detected, thow exception, but at the same time write value detection info of each time to log file (this requires set ascend log level to info or debug)
     - Currently, this feature only supports Atlas A2 training series products, and only detects abnormal feature value that occur during the training of Transformer class models with bfloat16 data type

       Considering that the feature value range can not be known ahead, setting NPU_ASD_ENABLE to 1 is recommended to enable silent check, which prevents training interruption caused by false detection
   * - NPU_ASD_UPPER_THRESH
     - Controls the absolute numerical threshold for detection
     - String
     - The format is a pair of integers, where the first element controls the first-level absolute numerical threshold, and the second element controls the second-level absolute numerical threshold

       Decreasing the threshold can detect smaller fluctuations of abnormal data, increasing the detection rate, while increasing the threshold has the opposite effect

       By default, if this environment variable is not configured, `NPU_ASD_UPPER_THRESH=1000000,10000`
     -
   * - NPU_ASD_SIGMA_THRESH
     - Controls the relative numerical threshold for detection
     - String
     - The format is a pair of integers, where the first element controls the first-level relative numerical threshold, and the second element controls the second-level relative numerical threshold

       Decreasing the threshold can detect smaller fluctuations of abnormal data, increasing the detection rate, while increasing the threshold has the opposite effect

       By default, if this environment variable is not configured, `NPU_ASD_SIGMA_THRESH=100000,5000`
     -

For more information on feature value detection, see `Feature Value Detection <https://www.mindspore.cn/tutorials/en/master/debug/sdc.html>`_.


Third-party Library
-------------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - OPTION_PROTO_LIB_PATH
     - Specifies the RPOTO dependent library path.
     - String
     - File path, which can be a relative path or an absolute path.
     -
   * - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
     - Choose which language to use for the Protocol Buffers back-end implementation
     - String
     - "cpp": implementation using c++ backend

       "python": implementation using python back-end

       No setting or other value: implementation using python backend
     -
   * - ASCEND_OPP_PATH
     - OPP package installation path
     - String
     - Absolute path for OPP package installation
     - Required for Ascend AI processor environments only; the environment generally provided to the user is already configured and need not be concerned.
   * - ASCEND_AICPU_PATH
     - AICPU package installation path
     - String
     - Absolute path of the AICPU package installation
     - Required for Ascend AI processor environments only; the environment generally provided to the user is already configured and need not be concerned.
   * - ASCEND_CUSTOM_OPP_PATH
     - the installation path of the custom operator package
     - String
     - the absolute path of custom operator package installation
     - Required for Ascend AI processor environments only; the environment generally provided to the user is already configured and need not be concerned.
   * - ASCEND_TOOLKIT_PATH
     - TOOLKIT package installation path
     - String
     - the absolute path of custom operator package installation
     - Required for Ascend AI processor environments only; the environment generally provided to the user is already configured and need not be concerned.
   * - CUDA_HOME
     - CUDA installation path
     - String
     - Absolute path for CUDA package installation
     - Required for GPU environment only, generally no need to set. If multiple versions of CUDA are installed in the GPU environment, it is recommended to configure this environment variable in order to avoid confusion.
   * - MS_ENABLE_TFT
     - Enable `MindIO TFT <https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/ref/mindiottp/mindiotft001.html>`_ feature. Turn on TTP, UCE, TRE or ARF feature.
     - String
     - "{TTP:1,UCE:1,TRE:1,ARF:1}". TTP (Try To Persist): End of life CKPT, UCE (Uncorrectable Memory Error): Fault tolerance and recovery, TRE(Training Result Error): Restoring training result exceptions, ARF (Air Refuelling): Process level rescheduling and recovery feature. The four features can be enabled separately. If you only want to enable one of them, set the corresponding value to 1. Other values: MindIO TFT not turned on. (When using UCE or ARF, TTP is enabled by default. TRE can not be used with UCE or ARF feature.)
     - Graph mode can only be enabled on the Ascend backend and jit_level is set to "O0" or "O1".
   * - MS_TFT_IP
     - The IP address where the MindIO controller thread is located for processor connections.
     - String
     - The IP address.
     - Graph mode can only be enabled on the Ascend backend and jit_level is set to "O0" or "O1".
   * - MS_TFT_PORT
     - The MindIO controller thread binds to a port for processor connections.
     - Integer
     - Positive integer.
     - Graph mode can only be enabled on the Ascend backend and jit_level is set to "O0" or "O1".
   * - AITURBO
     - Optimize settings to enable accelerated usage of Huawei Cloud Storage.
     - String
     - "1": Optimize settings to enable accelerated usage of Huawei Cloud Storage. Other values: Disable accelerated usage of Huawei Cloud Storage. Default value: Empty.
     - Limited to the Huawei Cloud environment.

CANN
-----

For more information about CANN's environment variables, see `Ascend community <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/appdevg/aclpythondevg/aclpythondevg_02_0004.html>`_ . Please set the environment variables for CANN before `import mindspore` .

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_FORMAT_MODE
     - Set the default preferred format for Ascend and graph compilation grade O2 processes, with the entire network set to ND format
     - Integer
     - 1: The operator prioritizes the ND format.

       0: The operator prioritizes private formats.

       Default value: 1
     - This environment variable affects the choice of format for the operator, which has an impact on network execution performance and memory usage, and can be tested by setting this option to get a better choice of operator format in terms of performance and memory.

       Ascend AI processor environment and graph compilation grade O2 processes only.

Profiler
-----------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_PROFILER_OPTIONS
     - Set the Profiler's collection options
     - String
     - Configure the Profiler's collection options in the format of a JSON string. The following parameters are different from the instantiation Profiler method, but the value meanings are the same:

       activities (list, optional) - Set the devices for collecting performance data, multiple devices can be specified, default value: [CPU, NPU]. Possible values: [CPU], [NPU], [CPU, NPU].

       aic_metrics (str, optional) - Set the type of AI Core metrics. Default value: AicoreNone. Possible values: AicoreNone, ArithmeticUtilization, PipeUtilization, Memory, MemoryL0, ResourceConflictRatio, MemoryUB, L2Cache, MemoryAccess.

       profiler_level (str, optional) - Set the level of performance data collection. Default value: Level0. Possible values: Level0, Level1, Level2.

       Refer to other parameters, see `Description of MindSpore profile parameters <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.profiler.profile.html>`_.

     - This environment variable enables one of two ways to enable performance data collection with the input parameter instantiation Profiler method.
   * - PROFILING_MODE
     - Set the mode of CANN Profiling
     - String
     - true: Enable Profiling.

       false or not configured: Disable Profiling.

       dynamic: Dynamic collection of performance data model.
     - This environment variable is enabled by CANN Profiling. Profiler reads this environment variable for checking to avoid repeatedly enabling CANN Profiling. Users don't need to set this environment variable manually.

Dynamic Graph
--------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MS_PYNATIVE_CONFIG_STATIC_SHAPE
     - We use this switch to turn on graph distribution for calculating gradient in PyNative mode.
     - String
     - '1': Turn on graph distribution for calculating gradient.
       Not setting or other values: Turn off graph distribution.
     - If turn on, we use graph distribution

Build from source
------------------

.. list-table::
   :widths: 20 20 10 30 20
   :header-rows: 1

   * - Environment Variable
     - Function
     - Type
     - Value Range
     - Description
   * - MSLIBS_CACHE_PATH
     - Path where third-pary software built alongside MindSpore will be installed to, when building MindSpore from source code.
     - String
     - `~/.mslib`: Your expected path to install third-party software. Default value: None.
     - When this environment variable is set, MindSpore will install third-party software built from source code to this path, enabling these software to be shared throughout multiple compilations and save time spent builing them.
   * - MSLIBS_SERVER
     - Website where third-pary software' source code is downloaded from when building MindSpore from source code.
     - String
     - `tools.mindspore.cn`: Official MindSpore image for downloading third-party source code. Default value: None.
     - When this environment variable is set, MindSpore will download third-party source code from given address, avoiding issues due to unstable access to github.com, improving speed of downloading source code. This variable is inactive when `-S on` is set in your compile options.