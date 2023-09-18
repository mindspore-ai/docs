分布式并行启动方式
============================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_zh_cn/parallel/startup_method.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  dynamic_cluster
  ms_operator

多卡启动方式
----------------------

目前GPU、Ascend和CPU分别支持多种启动方式。主要有OpenMPI，动态组网和多进程启动三种方式。

- 多进程启动方式。用户需要启动和卡数对应的进程，以及配置rank_table表。可以访问 `运行脚本 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/parallel/train_ascend.html#运行脚本>`_，学习如何通过多进程方式启动多卡任务。
- OpenMPI。用户可以通过mpirun命令来启动运行脚本，此时用户需要提供host file文件。用户可以访问 `通过OpenMPI运行脚本 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/parallel/train_ascend.html#通过openmpi运行脚本>`_，学习如何使用OpenMPI启动多卡任务。
- 动态组网。MindSpore使用内部动态组网模块，无需对外部配置文件或者模块产生依赖，帮助实现多卡任务。用户可以访问 `动态组网启动章节 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/parallel/dynamic_cluster.html>`_，学习如何使用动态组网方式启动多卡任务。

+---------------+--------------+-----------------+-------------+
|               | GPU          | Ascend          | CPU         |
+===============+==============+=================+=============+
| OpenMPI       | 支持         | 支持            | 不支持      |
+---------------+--------------+-----------------+-------------+
| 多进程启动    | 不支持       | 支持            | 不支持      |
+---------------+--------------+-----------------+-------------+
| 动态组网      | 支持         | 支持            | 支持        |
+---------------+--------------+-----------------+-------------+

云上启动MindSpore分布式并行训练
-----------------------------------------

MindSpore Operator是MindSpore在Kubernetes上进行分布式训练的插件。CRD（Custom Resource Definition）中定义了Scheduler、PS、Worker三种角色，用户只需配置yaml文件，即可轻松实现分布式训练。

当前ms-operator支持普通单Worker训练、PS模式的单Worker训练以及自动并行（例如数据并行、模型并行等）的Scheduler、Worker启动。详细流程请参考  `ms-operator <https://gitee.com/mindspore/ms-operator>`_。
