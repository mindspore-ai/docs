分布式并行启动方式
============================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/parallel/startup_method.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  msrun_launcher
  dynamic_cluster
  mpirun
  rank_table

启动方式
----------------------

目前GPU、Ascend和CPU分别支持多种启动方式。主要有\ ``msrun``\、动态组网、\ ``mpirun``\和\ ``rank table``\四种方式：

- `msrun <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/msrun_launcher.html>`_： `msrun` 是动态组网的封装，允许用户使用单命令行指令在各节点拉起分布式任务，安装MindSpore后即可使用。此方式不依赖第三方库以及配置文件，具有容灾恢复功能，安全性较好，支持三种硬件平台。建议用户优先使用此种启动方式。
- `动态组网 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/dynamic_cluster.html>`_：动态组网需要用户手动拉起多进程以及导出环境变量，是 `msrun` 的具体实现，Parameter Server训练模式建议使用此方式，其余分布式场景建议使用 `msrun` 。
- `mpirun <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/mpirun.html>`_：此方式依赖开源库OpenMPI，启动命令简单，多机需要保证两两之间免密登录，推荐有OpenMPI使用经验的用户使用此种启动方式。
- `rank table <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/rank_table.html>`_：此方式需要在Ascend硬件平台使用，不依赖第三方库。手动配置rank_table文件后，就可以通过脚本启动并行程序，多机脚本一致，方便批量部署。

.. warning::
    `rank_table` 启动方式将在MindSpore 2.4版本废弃。

四种启动方式的硬件支持情况如下表：

+-------------------------+--------------+-----------------+-------------+
|                         | GPU          | Ascend          | CPU         |
+=========================+==============+=================+=============+
|  \ ``msrun``\           | 支持         | 支持            | 支持        |
+-------------------------+--------------+-----------------+-------------+
|    动态组网             | 支持         | 支持            | 支持        |
+-------------------------+--------------+-----------------+-------------+
|  \ ``mpirun``\          | 支持         | 支持            | 不支持      |
+-------------------------+--------------+-----------------+-------------+
|    \ ``rank table``\    | 不支持       | 支持            | 不支持      |
+-------------------------+--------------+-----------------+-------------+
