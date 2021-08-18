mindspore.export
=================

.. py:class:: mindspore.export(net, *inputs, file_name, file_format=""AIR"", **kwargs)

   将MindSpore网络模型导出为指定格式的文件。

   说明：

       当导出文件格式为AIR、ONNX时，单个张量的大小不能超过2GB。

   参数：

      - **net**(Cell) – MindSpore网络结构。

      - **inputs**(Tensor) – 网络的输入，如果网络有多个输入，需要将张量组成元组。

      - **file_name**(str) – 导出模型的文件名称。

      - **file_format**(str) – MindSpore目前支持导出“AIR”，“ONNX”和“MINDIR”格式的模型。

           - **AIR**: Ascend Intermediate Representation。一种Ascend模型的中间表示格式。推荐的输出文件后缀是“.air”。

           - **ONNX**: Open Neural Network eXchange。一种针对机器学习所设计的开放式的文件格式。推荐的输出文件后缀是“.onnx”。

           - **MINDIR**: MindSpore Native Intermediate Representation for Anf。一种MindSpore模型的中间表示格式。推荐的输出文件后缀是“.mindir”。

      - **kwargs**(dict) – 配置选项字典。

           - **quant_mode**(str): 如果网络是量化感知训练网络，那么quant_mode需要设置为“QUANT”，否则quant_mode需要设置为“NONQUANT”。

           - **mean**(float): 预处理后输入数据的平均值，用于量化网络的第一层。默认值：127.5。

           - **std_dev**(float): 预处理后输入数据的方差，用于量化网络的第一层。默认值：127.5。

           - **enc_key**(str): 用于加密的字节类型密钥，有效长度为16，24，或者32。

           - **enc_mode**(str): 指定加密模式，当设置enc_key时，选项有：“AES-GCM”，“AES-CBC”。默认值：“AES-GCM”。

  样例：

      .. code-block::

              >>> import numpy as np
              >>> from mindspore import export, Tensor
              >>>
              >>> net = LeNet()
              >>> input = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
              >>> export(net, Tensor(input), file_name='lenet', file_format='MINDIR')