﻿# API模板

## 类 Class

```text
.. py:class:: Tensor(default_input, dtype=None, init_with_data=True)

    Tensor简介。

    **参数：**

        - **参数1** (`Tensor`) – 参数1说明。
        - **参数2** (`int`) – 参数2说明。

    **返回：**

        返回说明。

    **样例：**

        .. code-block::

            >>> a = Tensor(np.ones((1, 6)))
            >>> b = a.reshape((2, 3))
            输出结果
```

## 属性 Property

```text
    .. py:property:: T

        返回转置后的张量。
```

## 方法 Method

```text
    .. py:method:: abs()

        方法说明。

        **返回：**

            返回说明。

        **支持平台：**

           ``Ascend``  ``GPU``  ``CPU``

        **样例：**

          .. code-block::

                >>> from mindspore import Tensor
                >>> a = Tensor([1.1, -2.1]).astype("float32")
                >>> output = a.abs()
                >>> print(output)
                [1.1 .2.1]
```

## 函数 Function

```text
    .. py:function:: name(参数)

        描述函数功能。

    **参数：**

        - **参数1** (`Tensor`) – 参数1说明。
        - **参数2** (`int`) – 参数2说明。

    **返回：**

        返回说明。
```

## 注意事项

1. 链接的用法：

    如果链接文本是网址，语法如下：

    ```text
    `tensor <https://www.gitee.com/mindspore/mindspore/blob/r1.5/mindspore/common/tensor.py>`_
    ```

    请注意，链接文本和 URL 的开头 < 之间必须有一个空格。

2. 表格的用法：

    表格必须包含多行，并且第一列单元格不能包含多行。如下所示：

    ```text
        =====  =====  =======
         A           B            A and B
        =====  =====  =======
        False       False      False
        True        False      False
        False       True       False
        True        True       True
        =====  =====  =======
    ```

3. 注意文中专有名词的正确书写，例如“Numpy”，“Python”，“MindSpore”等。

## 参考

- 有关rst的书写规则，请参考[rst入门](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)。
- 有关rst在Python领域的用法，请参考[rst在Python领域用法](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain)。