分布式并行启动方式
============================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/startup_method.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  dynamic_cluster
  mpirun
  rank_table

启动方式
----------------------

目前GPU、Ascend和CPU分别支持多种启动方式。主要有动态组网、\ ``mpirun``\和\ ``rank table``\三种方式：

- `动态组网 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dynamic_cluster.html>`_：此方式不依赖第三方库，具有容灾恢复功能，安全性好，支持三种硬件平台，建议用户优先使用此种启动方式。
- `mpirun <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/mpirun.html>`_：此方式依赖开源库OpenMPI，启动命令简单，多机需要保证两两之间免密登录，推荐有OpenMPI使用经验的用户使用此种启动方式。
- `rank table <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/rank_table.html>`_：此方式需要在Ascend硬件平台使用，不依赖第三方库。手动配置rank_table文件后，就可以通过脚本启动并行程序，多机脚本一致，方便批量部署。

三种启动方式的硬件支持情况如下表：

+-------------------------+--------------+-----------------+-------------+
|                         | GPU          | Ascend          | CPU         |
+=========================+==============+=================+=============+
|    动态组网             | 支持         | 支持            | 支持        |
+-------------------------+--------------+-----------------+-------------+
|  \ ``mpirun``\          | 支持         | 支持            | 不支持      |
+-------------------------+--------------+-----------------+-------------+
|    \ ``rank table``\    | 不支持       | 支持            | 不支持      |
+-------------------------+--------------+-----------------+-------------+
