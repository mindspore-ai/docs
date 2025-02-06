模型保存与加载
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_zh_cn/model_train/parallel/model_save_load.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  model_saving
  model_loading
  model_transformation

MindSpore中模型保存可以分为合并保存和非合并保存，模型的加载也可以分为完整加载和切片加载。若加载后的分布式切分策略或集群卡数改变，则需要对保存的checkpoint文件进行模型转换。详细请参考：

- `模型保存 <https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_saving.html>`_
- `模型加载 <https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_loading.html>`_
- `模型转换 <https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/model_transformation.html>`_
