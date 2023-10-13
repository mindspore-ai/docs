模型保存与加载
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_zh_cn/parallel/model_save_load.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  model_saving
  model_loading
  model_transformation

MindSpore中模型保存可以分为合并保存和非合并保存，模型的加载也可以分为完整加载和切片加载。若加载后的分布式切分策略或集群卡数改变，则需要对保存的checkpoint文件进行模型转换。详细请参考：

- `模型保存 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/model_saving.html>`_
- `模型加载 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/model_loading.html>`_
- `模型转换 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/model_transformation.html>`_
