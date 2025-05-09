Error Reporting Analysis
=========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/debug/error_analysis.rst
    :alt: View Source On Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  error_analysis/error_scenario_analysis
  error_analysis/minddata_debug
  error_analysis/mindrt_debug
  error_analysis/cann_error_cases
  error_analysis/mindir

Overview
---------

This section is used to introduce the various functional debugging capabilities provided by MindSpore for neural network developers and framework developers. Functional debugging refers to the debugging capabilities of developers in the process of developing neural network or framework functions, which is different from the debugging and tuning of performance and accuracy after the functions are implemented. The functional debugging is divided into network development debugging and framework development debugging from different purposes of use. Network development debugging is used to meet the debugging requirements of network developers (also known as users) for error debugging, control and observation of network execution during neural network development, and framework development debugging is used to meet the debugging requirements of framework developers.

- network development debugging: The functional debugging capabilities for neural network developers can be divided into network error debugging and network execution debugging.

  - network error debugging: Provide error diagnosis and debugging capabilities when network errors are reported, e.g., error description, debugging with PyNative.
  - network execution debugging: Provide the observation and execution control capabilities during normal network execution, e.g. callback, hook.

- Framework development debugging: Provide functional debugging capabilities for MindSpore framework developers, such as logging and RDR (run data saving).

Network development debugging and framework development debugging are only distinguished from a more applicable perspective, not a strict functional division. Network developers can also use the framework development debugging function to debug problems, and vice versa.

Network Error Debugging
------------------------

Network error debugging is to solve the problem of error reporting during network training or inference. By understanding the meaning of the error message, hypothesize the cause of the problem, and use debugging methods to verify the hypothesis. Network error debugging is usually a process of multiple hypothesis and verification cycles, which includes two parts: error analysis and debug positioning. Error analysis is the process of obtaining the content of reported errors, understanding the description and analyzing their causes, mainly including information summarization, error analysis and error retrieval. Debug positioning is the process of selecting a suitable debugging strategy for the problem scenario and verifying the assumptions of the reported error, which mainly includes strategy selection, fault recurrence, and debug verification.

Error Analysis
~~~~~~~~~~~~~~~

Error analysis is the process of obtaining the content of error reports, understanding their description and analyzing their causes.

Information Summarization
^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step in network error debugging is to summarize information, where information summarization refers to categorizing the information obtained and understanding its meaning to provide a basis for error analysis. Generally, several types of information needs to be obtained when an error is reproted:

1. Information about the environment in which the error is reported, including: OS type and version, MindSpore version, execution mode (dynamic graph mode OR static graph mode), device information (x86 or ARM, Ascend or GPU)
2. Error description information, including: error type, error description, error stacks, etc.
3. If suspect a frame problem, you need to get the log information printed by the frame.

Understanding the meaning of error description information plays an important role in problem analysis, and the following will introduce how to read and understand MindSpore error messages.

MindSpore error messages are processed by using Python
Traceback processing, including Python stack information, error types and error descriptions, error messages related to networkdevelopers, and error messages related to framework developers. As shown in the following figure:

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/debug/images/graph_errmsg.png

-  Python stack information:

   As shown in figure ①, you can see the Python stack calling relationship from top to bottom.

-  Error type and error description:

   As shown in figure ②, the error type is \ ``TypeError``\, that is, parameter type error. The error description is the cause of the error. The number of parameters in the function is not correct, and the number of input parameters is required to be 2, but the actual number of parameters provided is 3.

-  Error messages related to networkdevelopers

   As shown in figure ③, it contains \ ``The Traceback of Net Construct Code``\ and other error messages related to network developers. \ ``The Traceback of Net Construct Code``\ is the C++ back-end compilation error location mapped to the Python front-end code call stack, printed in reverse order, corresponding to the \ ``construct``\ function or \ ``@ms_function``\ decorator-modified function in the neural network.

-  Error messages related to framework developers

   As shown in figure ④, it contains C++ stack information and other error reporting information related to framework developers, marked by \ ``For framework developers``\. You can set the environment variable \ ``export MS_EXCEPTION_DISPLAY_LEVEL=1``\ to hide the error message related to the framework developer. The default value of this environment variable is 0, indicating that error messages related to the framework developer are displayed by default.

Error Analysis
^^^^^^^^^^^^^^^^

Performing error analysis is an important step in network error debugging. Error analysis refers to the analysis of error causes based on various information obtained from the network and framework (e.g., error messages, network codes, and other information) to infer the possible causes of errors.

The general process of MindSpore network training is data loading and processing, network construction and training. In distributed parallel scenario, it also includes distributed parallel mode configuration. The error analysis of network error reporting usually includes the following steps:

1) Based on the error message, identify which problem scenario it is, such as data loading and processing problem scenario, network construction and training problem scenario, or distributed parallelism problem scenario. Usually, a distinction can be made by using the error message related to the network developer.
2) Analyze the problem scenarios and further identify which type of problem it is under that problem scenario. For example, the data loading and processing problem scenario includes three types: data preparation problems, data loading problems and data enhancement problems. Usually, the distinction needs to be made based on the type of error reported and the description of the error reported.
3) Analyze the location where the error is reported based on the Python call stack and the error information. In dynamic graph mode, it is easier to determine the location of the code error. In the static graph mode, you need to analyze the location of the error report according to the error message "The Traceback of Net Construct Code" part of the error message.
4) Based on possible error problem scenarios and types, hypothesize the possible causes of the error problem.

Please refer to error analysis for details on how to perform `error analysis <https://www.mindspore.cn/tutorials/en/master/debug/error_analysis/error_scenario_analysis.html>`_ based on different scenarios.

Error Search
^^^^^^^^^^^^^

Based on the error message and the location of the error code, combined with the common errors and possible causes in different scenarios, we can generally solve the common problems of parameter configuration errors, API interface usage errors, static graph syntax errors, etc. For more complex error reporting analysis, you can first try to search for cases. Of course, to improve the efficiency of problem solving, when you encounter an error reporting problem, you can directly perform an error search.

-  FAQ

   MindSpore provides FAQ for common error reporting issues, including data processing, compilation execution, distributed parallelism and other scenarios. Based on the problem scenarios derived from the error analysis, you can search for problems by using the error description information.

   The search address is as follows: \ `FAQ <https://www.mindspore.cn/docs/en/master/faq/installation.html>`__\ .

-  Error reporting case

   To cover more error reporting scenarios and improve users' problem solving ability, MindSpore in Forum provides common typical error reporting cases and introduces error analysis and solution methods. The prerequisite for error search is to select the appropriate search keywords. Usually, the search keywords are selected in the Error Report Type and Error Report Description sections of the error message. Usually, when searching in Cloud Forum, you can use the structure of subject + predicate + object, verb + object, subject + tense + epithet to search. For example, there is the following error message reported:

   .. code:: cpp

      Unexpected error. Invalid file, DB file can not match file

      Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 997).

      'self.val' should be initialized as a 'Parameter' type

   You can choose "DB file can not match file", "Exceed function call depth limit", and "should be initialized as a Parameter" as key words.

   The search address is as follows: \ `Error reporting case <https://www.hiascend.com/developer/blog/details/0229108045633055169>`__\ .

-  Community Issue

   In addition, MindSpore open source community has a lot of issues feedbacked by developers, involving network development error reporting, framework failure and many other issues. Users can search for similar problems using, for example, network name, error reporting content keywords. The keyword selection can refer to the error reporting case.

   The search address is as follows: \ `MindSpore
   Issues <https://www.hiascend.com/developer/blog/details/0229108045633055169>`__\ .

Debug Positioning
~~~~~~~~~~~~~~~~~~

Strategy Selection
^^^^^^^^^^^^^^^^^^^

-  Static to Dynamic Debugging Strategy

   Dynamic graph mode is a better debugging execution mode.
   Set the dynamic graph mode way: \ ``set_context(mode=mindspore.PYNATIVE_MODE)``\ .
   The program in dynamic graph mode is executed line by line according to the order in which the code is written, avoiding the back-end and front-end compilation optimization in static graph mode and ensuring uniform user code and execution logic. Dynamic diagrams are executed line by line to avoid the black box execution of the whole diagram sinking in diagram mode, which is more convenient to print the execution results and track the execution process.

-  Asynchronous-to-synchronous Debugging Strategy

   Dynamic diagram mode uses asynchronous execution by default in order to improve the efficiency of dynamic diagram execution, and error information are displayed at the last stage of execution. In Figure 3, you can see that the asynchronous execution method of error reporting will have alarm messages that interfere with the error reporting analysis.

   MindSpore provides a way to switch synchronous execution by setting \ ``set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)`` to switch to synchronous execution. If the operator execution error occurs, the task terminates directly and displays the current error message. For details, see \ `PyNative Synchronous Execution <https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html>`__\ .

-  Dichotomy Strategy

   Simplifying the problem scenario is an effective way to improve debugging efficiency. Based on the error message, it is usually a reliable way to confirm the scope of the error reporting problem and eliminate unnecessary influencing factors. In the case where the scope of error reporting cannot be accurately determined, a dichotomous approach can be attempted. For example, if the network calculation process contains nan values, the dichotomy method can be used to debug the data processing module and the network calculation module separately to verify whether the data input to the network calculation contains nan values and to confirm whether the outliers are introduced by the data or generated during the calculation process.

-  Deductive Inference Strategy

   Deductive inference is the process of deducing the cause of a problem and further verifying the conclusion. MindSpore error debugging is a step-by-step reverse inference based on the causal chain of problem propagation to locate the root cause of the problem. For example, the execution of MindSpore operator reports an error problem, which is directly caused by the fact that the input data of the operator contains illegal values, and the illegal values are derived from the computation of the previous operator, so we need to analyze whether the input data and computation process of the previous operator are correct. If there is a problem with the computation process of the previous operator, i.e., the scope of the problem is confirmed, if there are also illegal values in the input data of the previous operator, it is necessary to continue analyzing the previous operator until the root cause of the problem is found.

Problem Recurrence
^^^^^^^^^^^^^^^^^^^

Stable problem recurrence is a prerequisite for network debugging and a condition to verify whether the problem is completely solved. The network training process introduces randomness due to random initialization of network parameters, and different input data, which can easily cause inconsistent running results or error reporting locations.

Debugging Verification
^^^^^^^^^^^^^^^^^^^^^^^^

-  Dynamic graph debugging

   Due to the line-by-line code execution, single-step debugging, breakpoint debugging and process tracing can be performed by using the debugging tool pdb.
   Debugging Steps:

   1. insert import pdb; pdb.set_trace() before the code you want to debug to enable pdb debugging.
   2. run the .py file normally. The following similar result will appear in the terminal, debug by entering the corresponding pdb command after the (Pdb).
   3. Enter commands such as l and p in pdb interactive mode to view the corresponding code and variables, and then troubleshoot the related problems.

-  Static graph debugging

   1. ops.print\_

      In static graph mode, MindSpore provides `ops.print_ <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.print_.html>`_ interface to print Tensor information or string information in the computational graph.
      The outputs are printed to screen by default, and it can also be saved in a file.

   2. Debugger

      The Debugger can be used for reporting errors during the execution phase of the computational graph. Using the debugger, the following can be implemented.

      1. View the output of the graph node in the debugger interface in conjunction with the computational graph.

      2. Set monitoring points to monitor for training anomalies (such as check tensor overflows) and to track the cause of errors when they occur.

      3. View changes in parameters such as weights.

      4. Check the correspondence between diagram nodes and source code.

Network Execution Debugging
----------------------------

Network execution debugging is the corresponding debugging capability provided by MindSpore to meet the demands of network developers for network execution process observation and execution control, which can be divided into network execution observation and network execution control.

-  Network execution observation: Obtain the internal state or data of the network to observe the network execution information during the network execution, for example, visualization of training process, intermediate file (i.e. IR) saving function.
-  Network execution control: Perform specific actions during specific periods of network execution, for example, monitor loss, save model parameters, terminate training tasks early.

+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
| Function classification   | Main debugging functions                 | Description of use                                                                                          | Detailed introduction                                 |
+===========================+==========================================+=============================================================================================================+=======================================================+
| Execution observation     | Visualization of the training process    | The scalar, image, computation graph, training optimization process                                         |                                                       |
|                           |                                          | and model superparameter information during the training process are recorded in a file.                    |                                                       |
|                           |                                          |                                                                                                             |                                                       |
|                           |                                          |                                                                                                             |                                                       |
|                           |                                          |                                                                                                             |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | Training traceability and comparison     | The model traceability, data traceability, and comparison dashboard allow users to observe                  |                                                       |
|                           |                                          | different scalar trend graphs to identify problems and then use the traceability function                   |                                                       |
|                           |                                          | to locate the cause of the problem, providing users with the ability to efficiently                         |                                                       |
|                           |                                          | tune the data augmentation and deep neural networks.                                                        |                                                       |
|                           |                                          |                                                                                                             |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | metrics                                  | When the training is finished,                                                                              |                                                       |
|                           |                                          | metrics can be used to evaluate the training results.                                                       |                                                       |
|                           |                                          | A variety of metrics are provided for evaluation,                                                           |                                                       |
|                           |                                          | such as: accuracy, loss, preci sion, recall, F1.                                                            |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | print\_ interface                        | The print\_ interface prints out the Tensor or                                                              | `print\_ interface introduction <https://www.minds    |
|                           |                                          | string information entered by the user.                                                                     | pore.cn/docs/en/master/api_python/ops/                |
|                           |                                          |                                                                                                             | /mindspore.ops.print_.html>`_                         |
|                           |                                          |                                                                                                             |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | Intermediate file saving                 | Used to save the intermediate files generated                                                               | `Reading IR <https://www.mindspore.cn/tutorials       |
|                           |                                          | during the diagram compilation process, which we call IR files, to support                                  | /en/master/debug/error_analysis                       |
|                           |                                          | the diagnosis of problems related to diagram structure and diagram information.                             | /mindir.html>`_                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | Data Dump                                | When training the network, if the training result deviates from the expectation,                            | `Dump function debugging <https://www.mindspore.cn/   |
|                           |                                          | the operator input and output data are saved for debugging by the Du mp function.                           | tutorials/en/master/debug/dump.html>`_                |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
| Execution control         | Callback                                 | Users can use callback functions to perform specific actions                                                |                                                       |
|                           |                                          | at specific times or to observe network information                                                         |                                                       |
|                           |                                          | during training, e.g., save model parameters, monitor loss,                                                 |                                                       |
|                           |                                          | dynamically adjust parameters, terminate training tasks early.                                              |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | Hook                                     | The Hook function in pynative mode captures the input and output data                                       | `Hook function <https://www.mindspore.cn/             |
|                           |                                          | and the backward gradient of the middle layer operator.                                                     | tutorials/en/master/custom_program/                   |
|                           |                                          | Four forms of Hook functions are available:                                                                 | hook_program.html>`_                                  |
|                           |                                          | HookBackward operator and register_forward_pre_hook, register_forward_hook,                                 |                                                       |
|                           |                                          | and register_backward_hook functions                                                                        |                                                       |
|                           |                                          | registered on the Cell object.                                                                              |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+
|                           | Synchronous execution                    | In dynamic graph mode, operators are executed asynchronously                                                | `Synchronized execution of dynamic graph <https://    |
|                           |                                          | on the device to improve performance,                                                                       | www.mindspore.cn/tutorials/en/                        |
|                           |                                          | so operator execution errors may be displayed at the end of program execution.                              | master/beginner/accelerate_with_static_graph.html     |
|                           |                                          | In this case, MindSpore provides a synchronous execution setting                                            | #dynamic-graph-mode>`_                                |
|                           |                                          | to control whether the arithmetic is executed asynchronously on the device.                                 |                                                       |
+---------------------------+------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------+

Framework Development Debugging
--------------------------------

MindSpore provides framework developers with rich debugging tools. Debugging features cover the framework's execution process, the framework's execution data, the framework's special control, for example: the log can record the framework's execution process, and the RDR can record the framework's key state information, memory reuse control.

+-------------------------+--------------------------+---------------------------------------------------------------+-------------------------------------------------------+
| Function classification | Main debugging functions |  Description of use                                           | Detailed introduction                                 |
+=========================+==========================+===============================================================+=======================================================+
| Process records         | Logs                     | used to record information at each stage of the framework     | `Log-related environment variables and configurations |
|                         |                          | implementation to provide information for understanding       | <https://www.mindspore.cn/docs/en/master              |
|                         |                          | the framework implementation process or for problem diagnosis.| /api_python/env_var_list.html>`_                      |
|                         |                          |                                                               |                                                       |
+-------------------------+--------------------------+---------------------------------------------------------------+-------------------------------------------------------+
| Data records            | RDR                      | Running Data Recorder (RDR) provides the ability              |                                                       |
|                         |                          | to record framework execution status data                     |                                                       |
|                         |                          | while the training program is running.                        |                                                       |
|                         |                          | It can also save key frame state data, such as IR,            |                                                       |
|                         |                          | graph execution order,                                        |                                                       |
+-------------------------+--------------------------+---------------------------------------------------------------+-------------------------------------------------------+
| Specialized control     | Memory reuse             | Configure memory reuse on and off for troubleshooting         |                                                       |
|                         |                          | or debugging suspected problems related to memory reuse.      |                                                       |
|                         |                          |                                                               |                                                       |
+-------------------------+--------------------------+---------------------------------------------------------------+-------------------------------------------------------+
