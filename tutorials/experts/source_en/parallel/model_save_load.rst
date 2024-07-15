Model Saving and Loading
=========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/br_base/tutorials/experts/source_en/parallel/model_save_load.rst
    :alt: View Source on Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  model_saving
  model_loading
  model_transformation

Model saving in MindSpore can be categorized into merged and non-merged saving, and model loading can also be categorized into complete loading and sliced loading. If the distributed slicing strategy or cluster card is changed after loading, the saved checkpoint file needs to be model transformed. For details, please refer to:

- `Model Saving <https://www.mindspore.cn/tutorials/experts/en/br_base/parallel/model_saving.html>`_
- `Model Loading <https://www.mindspore.cn/tutorials/experts/en/br_base/parallel/model_loading.html>`_
- `Model Transformation <https://www.mindspore.cn/tutorials/experts/en/br_base/parallel/model_transformation.html>`_
