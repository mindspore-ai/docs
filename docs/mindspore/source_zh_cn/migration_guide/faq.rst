常见问题
===========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/migration_guide/faq.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  use_third_party_op

MindSpore官网提供了一份在使用MindSpore过程中的 `FAQ <https://mindspore.cn/docs/zh-CN/r2.3/faq/installation.html>`_ ，本章也整理了一下在迁移文档中提及的常见问题集解决方法。

- 数据处理

  **Q: 为什么在迭代数据的时候会报错：The actual amount of data read from generator xx is different from generator.len xx, you should adjust generator.len to make them match ？**

  A: 在定义可随机访问数据集时， __len__ 方法返回的结果一定要是真实的数据集大小，设置大了在getitem取值时会有越界问题。如数据集大小未确定，可以使用可迭代数据集，详见 `自定义数据集 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/beginner/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86>`_ 。


  **Q: 为什么在迭代数据的时候会报错：Invalid Python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names, the size of column_names is:xx and number of returned NumPy array is:xx ？**

  A: 这是因为GeneratorDataset的 column_names 参数指定的列名数量与 source 参数输出的数据数量不匹配。


  **Q: 使用 GeneratorDataset 或 map 进行加载/处理数据时，可能会因为语法错误、计算溢出等问题导致数据报错，如何进行排查和调试？**

  A: 观察报错栈信息，由报错栈信息大概定位到出错代码块，在出错的代码块附近添加打印或调试点，进一步调试。详细可参考 `数据处理调试方法一 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/minddata_debug.html#%E6%96%B9%E6%B3%951-%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%89%A7%E8%A1%8C%E5%87%BA%E9%94%99%E6%B7%BB%E5%8A%A0%E6%89%93%E5%8D%B0%E6%88%96%E8%B0%83%E8%AF%95%E7%82%B9%E5%88%B0%E4%BB%A3%E7%A0%81%E4%B8%AD%E8%B0%83%E8%AF%95>`_ 。


  **Q: 数据增强 map 操作出错，如何调试 map 操作中各个数据处理算子？**

  A: 可以通过单个算子执行的方式调试或者通过数据管道调试模式调试 map 操作。详细可参考 `数据处理调试方法二 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/minddata_debug.html#%E6%96%B9%E6%B3%952-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BAmap%E6%93%8D%E4%BD%9C%E5%87%BA%E9%94%99%E8%B0%83%E8%AF%95map%E6%93%8D%E4%BD%9C%E4%B8%AD%E5%90%84%E4%B8%AA%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E7%AE%97%E5%AD%90>`_ 。


  **Q: 在训练的时候，会获得非常多warning提示我们数据集性能较慢应该怎么处理？**

  A: 可以单独迭代数据集，查看每条数据的处理时间，以此判断数据集的性能如何。详细可参考 `数据处理调试方法三 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/minddata_debug.html#%E6%96%B9%E6%B3%953-%E6%B5%8B%E8%AF%95%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E7%9A%84%E6%80%A7%E8%83%BD>`_ 。 


  **Q: 在对数据进行处理的过程中，如果因为计算错误、数值溢出等因素，产生了异常的结果数值，从而导致训练网络时算子计算溢出、权重更新异常等问题该怎么排查？**

  A: 关闭混洗，固定随机种子，确保可重现性，然后利用NumPy等工具快速校验结果。详细可参考 `数据处理调试方法四 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/minddata_debug.html#%E6%96%B9%E6%B3%954-%E6%A3%80%E6%9F%A5%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E4%B8%AD%E7%9A%84%E5%BC%82%E5%B8%B8%E6%95%B0%E6%8D%AE>`_ 。


  更多数据处理常见问题请参考 `数据处理常见问题分析 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/minddata_debug.html#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%88%86%E6%9E%90>`_ 以及迁移中的数据处理差异请参考 `MindSpore和PyTorch的数据处理差异 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/dataset.html#数据处理差异对比>`_ 。

- 网络脚本

  `API映射及缺失API处理策略 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/analysis_and_preparation.html#分析api满足度>`_

  `动态shape分析 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/dynamic_shape.html>`_ 及 `规避方案 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/model_and_cell.html#动态shape规避策略>`_

  `稀疏特性规避方案 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/sparsity.html>`_

  `静态图常见语法限制及处理策略 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/model_and_cell.html#动态图与静态图>`_ 

  `MindSpore网络编写注意事项 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/model_development.html#mindspore网络编写注意事项>`_
	
  `基于自定义算子接口调用第三方算子库 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/use_third_party_op.html>`_

  `PyTorch模型转换MindSpore模型的方法 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/sample_code.html#模型验证>`_

- 网络调试

  `功能调试 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/debug_and_tune.html#功能调试>`_

  `精度调试 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/debug_and_tune.html#精度调试>`_

  `性能调优 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/debug_and_tune.html#性能调优>`_

