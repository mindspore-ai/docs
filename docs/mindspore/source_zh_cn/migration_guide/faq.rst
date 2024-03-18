常见问题
===========

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/migration_guide/faq.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  use_third_party_op

MindSpore官网提供了一份在使用MindSpore过程中的 `FAQ <https://mindspore.cn/docs/zh-CN/r2.3/faq/installation.html>`_ ，本章也整理了一下在迁移文档中提及的常见问题及解决方法。

- 环境准备

  **Q: 如何搭建MindSpore环境？**

  A: MindSpore目前支持在昇腾、GPU、CPU等多种设备上运行，但在安装过程中需要注意选择配套的硬件平台、操作系统、Python版本，否则会出现很多不可预测的报错。详细可参考 `安装指导 <https://www.mindspore.cn/install/>`_ 。

  更多环境准备常见问题请参考 `环境准备常见问题分析 <https://www.mindspore.cn/docs/zh-CN/r2.3/faq/installation.html>`_ 。

- 模型分析与准备

  **Q: 如何查看MindSpore对迁移代码中的API支持程度？**

  A: 可以使用API自动扫描工具MindSpore Dev Toolkit（推荐），或手动查询API映射表进行分析。详细可参考 `分析API满足度 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/analysis_and_preparation.html#%E5%88%86%E6%9E%90api%E6%BB%A1%E8%B6%B3%E5%BA%A6>`_ 。

- 数据处理

  **Q: 怎么将PyTorch的`dataset`转换成MindSpore的`dataset`？**

  A: MindSpore和PyTorch的自定义数据集逻辑是比较类似的，首先需要用户先定义一个自己的 `dataset` 类，该类负责定义 `__init__` 、 `__getitem__` 、 `__len__` 来读取自己的数据集，然后将该类实例化为一个对象（如: `dataset/dataset_generator` ），最后将这个实例化对象传入 `GeneratorDataset` (mindspore用法)/ `DataLoader` (pytorch用法)，至此即可以完成自定义数据集加载了。

  而MindSpore在 `GeneratorDataset` 的基础上提供了进一步的 `map` -> `batch` 操作，可以很方便地让用户在 `map` 内添加一些其他的自定义操作，并将其 `batch` 起来。

  对应的MindSpore的自定义数据集加载如下:

  .. code-block:: python

      # 1 Data enhancement,shuffle,sampler.
      class Mydata:
          def __init__(self):
              np.random.seed(58)
              self.__data = np.random.sample((5, 2))
              self.__label = np.random.sample((5, 1))
          def __getitem__(self, index):
              return (self.__data[index], self.__label[index])
          def __len__(self):
              return len(self.__data)
      dataset_generator = Mydata()
      dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
      # 2 Customized data enhancement
      dataset = dataset.map(operations=pyFunc, {other_params})
      # 3 batch
      dataset = dataset.batch(batch_size, drop_remainder=True)

  **Q: 为什么在迭代数据的时候会报错：“The actual amount of data read from generator xx is different from generator.len xx, you should adjust generator.len to make them match” ？**

  A: 在定义可随机访问数据集时， `__len__` 方法返回的结果一定要是真实的数据集大小，设置大了在 `__getitem__`取值时会有越界问题。如数据集大小未确定，可以使用可迭代数据集，详见 `自定义数据集 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/beginner/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86>`_ 。


  **Q: 为什么在迭代数据的时候会报错：“Invalid Python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names, the size of column_names is:xx and number of returned NumPy array is:xx” ？**

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

- 梯度求导

  **Q: 如何自己实现算子的反向计算？**

  A: MindSpore提供了自动的梯度求导接口，该功能对用户屏蔽了大量的求导细节和过程。但如果有某些特殊场景，用户需要手动控制其反向的计算，用户也可以通过Cell.bprop接口对其反向进行定义。详细可参考 `自定义Cell反向 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/modules/layer.html#%E8%87%AA%E5%AE%9A%E4%B9%89cell%E5%8F%8D%E5%90%91>`_ 。

  **Q: 如何处理梯度溢出造成训练不稳定的问题？**

  A: 网络溢出一般表现为loss Nan/INF，loss突然变得很大等。MindSpore提供 `dump数据 <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/debug/dump.html>`_ 获取到溢出算子信息。当网络中出现梯度下溢时，可使用loss scale配套梯度求导使用，详细可参考 `loss scale <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/gradient.html#loss-scale>`_ ；当网络出现梯度爆炸时，可考虑添加梯度裁剪，详细可参考 `梯度裁剪 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/model_development/gradient.html#%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA>`_ 。

- 调试调优

  **Q: 请问想加载PyTorch预训练好的模型用于MindSpore模型finetune有什么方法？**

  A: 需要把PyTorch和MindSpore的参数进行一一对应，因为网络定义的灵活性，所以没办法提供统一的转化脚本。

  一般情况下，CheckPoint文件中保存的就是参数名和参数值，调用相应框架的读取接口后，获取到参数名和数值后，按照MindSpore格式，构建出对象，就可以直接调用MindSpore接口保存成MindSpore格式的CheckPoint文件了。

  其中主要的工作量为对比不同框架间的parameter名称，做到两个框架的网络中所有parameter name一一对应(可以使用一个map进行映射)，下面代码的逻辑转化parameter格式，不包括对应parameter name。

  .. code-block:: python

      import torch
      import mindspore as ms

      def pytorch2mindspore(default_file = 'torch_resnet.pth'):
          # read pth file
          par_dict = torch.load(default_file)['state_dict']
          params_list = []
          for name in par_dict:
              param_dict = {}
              parameter = par_dict[name]
              param_dict['name'] = name
              param_dict['data'] = ms.Tensor(parameter.numpy())
              params_list.append(param_dict)
          ms.save_checkpoint(params_list,  'ms_resnet.ckpt')

  **Q: loss不收敛或精度不达标，该怎么定位？**

  A: 精度不达标一般体现在loss不收敛上。但是有很多复杂的原因可导致精度达不到预期，定位难度较大。这里提供几个指导链接供用户逐一排查问题。

  `MindSpore模型精度调优实战（一）精度问题的常见现象、原因和简要调优思路 <https://www.hiascend.com/forum/thread-0215121673876901029-1-1.html>`_ 。

  `MindSpore模型精度调优实战（二）精度调试调优思路 <https://www.hiascend.com/forum/thread-0235121941309178031-1-1.html>`_ 。

  `MindSpore模型精度调优实战（三）常见精度问题简介 <https://www.hiascend.com/forum/thread-0235121941523411032-1-1.html>`_ 。

  更多调试调优常见问题请参考 `调优常见问题及解决办法 <https://www.mindspore.cn/docs/zh-CN/r2.3/migration_guide/debug_and_tune.html#%E8%B0%83%E4%BC%98%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%8F%8A%E8%A7%A3%E5%86%B3%E5%8A%9E%E6%B3%95>`_ 。

  **Q: 模型训练过程中，第一个step耗时很长，该怎么优化？**

  A: 模型训练过程中，第一个step包含网络编译时长，如果想要优化第一个step的性能，可分析模型编译是否能进行优化。详细可参考 `静态图网络编译性能优化 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/static_graph_expert_programming.html>`_ 。

  **Q: 模型训练过程中，非首个step耗时很长，该怎么优化？**

  A: 模型训练过程中，非首个step的耗时包括迭代间隙、前反向计算和迭代拖尾，如果想要优化非首step的性能，需要先获取网络的迭代轨迹，再分析哪部分是性能瓶颈，最近进行性能优化。
     
  详细可参考 `性能调优指南 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.2/performance_tuning_guide.html>`_ ；和 `性能调试案例 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.2/performance_optimization.html>`_ 。

  **Q: 加载标杆权重进行模型推理验证正向流程时，有warning警告显示权重未加载成功，该如何解决？**

  A: load_checkpoint过程中，如果有权重未加载上，MindSpore会给出warning提示，一般加载失败有两种原因：1、权重名称对不上；2、权重在网络中缺失。

  如果权重名称对不上，需要打印MindSpore的权重名称和标杆的权重名称，看是否MindSpore的权重名称多了backbone或network等前缀，如果是，检查MindSpore在初始化 `Cell <https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.Cell.html>`_ 时是否加上auto_prefix=False。

  如果权重名称缺失，需要分析是否合理，如果合理，可忽略告警提示，如果不合理，需要分析网络定义是否错误，进行定位修改。

  **Q: 迁移过程使用PyNative进行调测，流程成功，切换成Graph模式，为什么会出现一堆的报错？**

  A: PyNative模式下模型进行推理的行为与一般Python代码无异。但是切换成Graph模式时，MindSpore通过源码转换的方式，将Python的源码转换成中间表达IR（Intermediate Representation），并在此基础上对IR图进行优化，最终在硬件设备上执行优化后的图。

  而这一步操作中MindSpore目前还未能支持完整的Python语法全集，所以construct函数的编写会存在部分限制。
  
  如：PyNative模式下可直接判断某个Tensor值是否为0，但切换成Graph模式则会报错不支持。

  .. code-block:: python

      if response == 0:
          return loss
      return loss/response

  遇到类似情况，可将代码修改为：

  .. code-block:: python

      response_gt = max(response, ms.Tensor(1))
      loss = loss/response_gt
      return loss

  详细可参考 `静态图语法支持 <https://www.mindspore.cn/docs/zh-CN/r2.3/note/static_graph_syntax_support.html>`_ 。

  **Q: 训练过程中出现报错：“RuntimeError: Launch kernel failed, name:Default/...” 怎么办？**

  A: 这类报错一般是MindSpore不支持某个算子，可能需要用户自己实现该算子。详细可参考 `PyTorch与MindSpore API映射表 <https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html>`_ 。

  **Q: PyNative动态图迁移过程中出现报错，该怎么有效地定位到报错原因？**

  A: 如果遇到动态图问题，可以设置mindspore.set_context(pynative_synchronize=True)查看报错栈协助定位。详细可参考 `pynative_synchronize说明 <https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.set_context.html?highlight=pynative_synchronize>`_ 。

  **Q: Graph模式静态图训练过程中出现报错，该怎么有效地定位到报错原因？**

  A: 引发静态图报错的原因很多，一般失败会有日志打印，如果不能直观地从日志中获取报错信息，可通过export GLOG_v=1指定日志级别获取更详细的报错信息进行分析。

  同时计算图编译发生报错时，会自动保存analyze_failed.ir文件，可帮助分析报错代码的位置。详细可参考 `静态图模式错误分析 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/error_scenario_analysis.html>`_ 。

  **Q: Graph模式静态图训练过程中出现Out Of Memory报错，怎么办？**

  A: 出现该报错可能有两个原因：1、资源被占用；2、显存不够。
     
  当资源被占用时，可通过pkill -9 python释放资源，再重新训练。
     
  当显存不够时，可尝试降低batch_size；分析内存查看是否通信算子太多导致整体内存复用率较低。
     
  详细可参考 `资源不够问题分析 <https://www.mindspore.cn/tutorials/zh-CN/r2.3/advanced/error_analysis/mindrt_debug.html#%E8%B5%84%E6%BA%90%E4%B8%8D%E8%B6%B3>`_ 。

  更多调优常见问题请参考 `执行问题 <https://www.mindspore.cn/docs/zh-CN/r2.3/faq/implement_problem.html>`_ 。
