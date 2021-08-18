mindspore.load_checkpoint
==========================

.. py:class:: mindspore.load_checkpoint(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode="AES-GCM")

  加载指定checkpoint文件的数据。

  参数：

      - **ckpt_file_name**(str) – checkpoint的文件名称。

      - **net**(Cell) – MindSpore网络结构。默认值：None。

      - **strict_load**(bool) – 是否将参数严格加载到网络中。如果是False, 它将根据相同的后缀名将参数字典中的参数加载到网络中，并会在精度不匹配时，进行精度转换。默认值：False。

      - **filter_prefix**(Union[str, list[str], tuple[str]]) – 以filter_prefix开头的参数将不会被加载。默认值：None。

      - **dec_key**(Union[None, bytes]) – 用于解密的字节类型密钥，如果值为None，则不需要解密。默认值：None。

      - **dec_mode**(str) – 该参数仅当dec_key不为None时有效。指定解密模式，目前支持“AES-GCM”和“AES-CBC”。默认值：“AES-GCM”。

  返回：

      字典，键是参数名称，值是Parameter类型的参数。

  抛出异常：

      **ValueError** – checkpoint文件不正确。

  样例：

      .. code-block::

              >>> from mindspore import load_checkpoint
              >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
              >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
              >>> print(param_dict["conv2.weight"])
              Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)



