mindspore.load_param_into_net
=========

.. py:class:: mindspore.load_param_into_net(net, parameter_dict, strict_load=False)

  将参数加载到网络中。

  参数：

      - **net**(Cell) – MindSpore网络结构。

      - **parameter_dict**(dict) – 参数字典。

      - **strict_load**(bool) – 是否将参数严格加载到网络中。如果是False, 它将以相同的后缀名将参数字典中的参数加载到网络中，并会在精度不匹配时，进行精度转换。默认值：False。

  返回：

      列表，未被加载到网络中的参数。

  抛出异常：

      TypeError – 如果参数不是Cell或者parameter_dict不是Parameter类型的字典。

  样例：

      .. code-block::

              >>> from mindspore import load_checkpoint, load_param_into_net
              >>>
              >>> net = Net()
              >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
              >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
              >>> param_not_load = load_param_into_net(net, param_dict)
              >>> print(param_not_load)
              ['conv1.weight']
