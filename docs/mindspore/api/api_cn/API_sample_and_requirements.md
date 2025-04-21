# 中文API格式规范

## 类 Class

```text
.. py:class:: mindspore.Tensor(default_input, dtype=None, init_with_data=True)

    Tensor简介。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int, 可选) – 参数2说明。默认值： ``None`` 。

          - **二级参数1** (int) – 二级参数1说明。（注意：二级参数需要和上面一级参数的“*”对齐。）
          - **二级参数2** (int) – 二级参数2说明。

        - **参数3** (bool, 可选) – 参数3说明。默认值： ``True`` 。

    返回：
        返回说明。

    输入：
        - **输入1** (Tensor) - 输入1描述。
        - **输入2** (Tensor) - 输入2描述。

    输出：
        - **输出1** (Tensor) - 输出1描述。
        - **输出2** (Tensor) - 输出2描述。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。
```

## 属性 Property

```text
.. py:method:: T
    :property:

    返回转置后的张量。
```

注意：相对于类要增加4空格缩进。

## 方法 Method

普通方法：

```text
.. py:method:: abs()

    方法说明。

    返回：
        返回说明。
```

注意：相对于类要增加4空格的缩进。

静态方法：

```text
.. py:method:: abs()
    :staticmethod:

    方法说明。

    返回：
        返回说明。
```

抽象方法：

```text
.. py:method:: abs()
    :abstractmethod:

    方法说明。
```

类方法：

```text
.. py:method:: abs()
    :classmethod:

    方法说明。

    返回：
        返回说明。
```

## 特殊方法 mindspore.Tensor

该接口与func接口同名时写法如下：

```text
.. py:method:: mindspore.Tensor.abs()

    详情请参考 :func:`mindspore.ops.abs`。
```

该接口是其他Tensor接口别名时写法如下：

```text
.. py:method:: mindspore.Tensor.absolute()

    :func:`mindspore.Tensor.abs` 的别名。
```

基于yaml自动生成的Tensor接口写法如下：

无重载函数：

```text
.. py:method:: mindspore.Tensor.gather(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。
```

有重载函数：

```text
.. py:method:: mindspore.Tensor.gather(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。

    .. py:method:: mindspore.Tensor.gather(参数)
        :noindex:

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。
```

## 函数 Function

无重载函数：

```text
.. py:function:: name(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。
```

有重载函数：

```text
.. py:function:: mindspore.mint.max(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。

    .. py:function:: mindspore.mint.max(参数)
        :noindex:

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。
```

## 样例特殊情况

注意：重载接口暂不支持特殊样例的写法！

英文样例内有note时，中文需要特殊处理（class、function、method相同写法，以function为例）：

```text
.. py:function:: name(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。

    样例：

    .. note::
        xxxx

```

英文注释内有教程样例时，中文需要特殊处理（class、function、method相同写法，以function为例）：

```text
.. py:function:: name(参数)

    描述函数功能。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int) – 参数2说明。

    返回：
        返回说明。

    异常：
        - **Error1** – 异常描述1。
        - **Error2** – 异常描述2。

    教程样例：
        - `自动混合精度 - 损失缩放
          <https://mindspore.cn/tutorials/zh-CN/br_base/beginner/mixed_precision.html#损失缩放>`_

```

如果英文既有教程样例，还有有note的样例，则教程样例的格式写在note样例的下面：

```text
    样例：

    .. note::
        xxxx

    教程样例：
        - `自动混合精度 - 损失缩放
          <https://mindspore.cn/tutorials/zh-CN/br_base/beginner/mixed_precision.html#损失缩放>`_

```

## Note

```text
.. note::
    此处描述具体需要注意的部分，前面需缩进四格。
```

## See also

```text
.. seealso::
    此处描述为对模块文档或外部文档的引用，前面需缩进四格。
```

## Warning

```text
.. warning::
    此处描述具体需要警告的部分，前面需缩进四格。
```

## 引入其他部分

```text

.. include:: {relative_file_path.rst}

```

引用其他`.rst`或`.txt`文件的内容，其中`{relative_file_path.rst}`为待引用文件的相对路径。

## 默认行为的描述方法

1. 参数有默认值，且不为None，直接表达为实际的默认值。

    中文：

    ```text
    默认值：``XXX``。
    ```

    英文：

    ```text
    Default: ``XXX``.
    ```

2. 参数有默认值，为None，需补充说明None的含义。

    中文：

    ```text
    默认值：``None``，表示框架默认设置为XXX。
    ```

    英文：

    ```text
    Default: ``None``, indicates that the default value in the framework is XXX.
    ```

3. 参数没有默认值，框架有默认值，在接口描述里写框架默认行为。

    中文：

    ```text
    框架默认***。
    ```

    英文：

    ```text
    The framework *** by default.
    ```

4. 参数没有默认值，框架也没有默认值，不需要任何说明。

## 内容注意事项

1. 类（class，如mindspore.nn模块）文档中可能包含参数、输入、输出、异常；函数（function，如mindspore.ops模块）和方法（method，如mindspore.Tensor中的方法）文档中可能包含参数、返回、异常。

2. 参数模块：

    - 参数顺序需与API定义中的参数顺序保持一致。
    - 如果是可选参数，需在参数数据类型后面增加“可选”字样，在结尾说明默认值并根据需要解释其含义。
    - API定义中若含有 ``*`` 项，则 ``*`` 之后的参数需单独写至“关键字参数”模块中，写法与其他参数一致。

3. 异常模块：将同类异常放在一起。

4. 维度描述：描述维度时，需使用文字方式表示数字，如零维数组、一维数组、二维数组、三维数组等。

## 其他格式注意事项

1. 链接的用法：

    如果链接文本是网址，语法如下：

    ```text
    `链接文本 <超链接URL>`_

    例：

    请参考 `tensor <https://www.gitee.com/mindspore/mindspore/blob/br_base/mindspore/python/mindspore/common/tensor.py>`_ 。
    ```

    请注意，链接文本和 URL 的开头 < 之间必须有一个空格，且整体的前后需要有空格。

2. 表格的用法：

    表格必须包含多行，并且第一列单元格不能包含多行。如下所示：

    ```text
    =====  =====  =======
     A      B     A and B
    =====  =====  =======
    False  False   False
    True   False   False
    False  True    False
    True   True    True
    =====  =====  =======
    ```

3. 注意文中专有名词的正确书写，例如“NumPy”，“Python”，“MindSpore”等。

4. 以下词请保持英文原词，无需翻译成中文：

    a. 参数名称，例如“Args”里面的参数解释。

    b. 数据类型，例如：Number, String, Tuple, List, Set, Dictionary, int, float, bool, complex, 等等。

    c. 报错和默认值，例如：RuntimeError, ValueError, None, True, False。

    d. 一些专有名词，例如：shape, Tensor, Ascend, checkpoint 以及加“ ”，\` \`, ‘ ’等特殊表示的词。

    e. Raises翻译为“异常”。

    f. Keyword Args翻译为“关键字参数”。

5. 常用特殊字符标记：

    - 星号： 写法为： \*text\* 是强调 (斜体)，效果为：*text*
    - 双星号： 写法为：\*\*text\*\* 重点强调 (加粗)，效果为：**text**
    - 反引号： 写法为：\`\`text\`\` 代码样式，效果为：``text``

    标记需注意的一些限制：

    - 不能相互嵌套，例如：\`\`\* text\*\`\` 是错误的。
    - 内容中间不能有空格： 这样写\`\`  text \`\` 是错误的。
    - 特殊字符标记的前后需要用空格隔开，例如：\`init\` 的值可以是 \`ones\` 或 \`zeros\` 等。
    - 如果内容需要特殊字符分隔，使用反斜杠转义。

6. 中文API文档中不需要手写“支持平台”和“样例”部分。
7. 当前rst文档作为整个API的文档页面时，需要添加标题，且标题名称为当前接口的全名；当前rst文档作为其他rst文档的引用时，不需要添加标题。
8. `mindspore.train`模块下的接口，需要将每个API的文档放于`train`目录下，并在`mindspore.train.rst`文件中通过`.. include::`的写法将接口引入过来；其他模块下的接口结构参考英文的API文档。

9. 引用类/函数的用法

    ```text
    :class:`类的全称`

    :func:`函数的全称`

    :attr:`属性的全称`
    ```

    a. 引用其他类的内容，类的全称类似于mindspore.train.Metric，包含一级二级类别的名称。引用其他函数的内容，函数的全称类似于mindspore.ops.dropout。

    b. 如果文档内不想写全称，可写成 :class:`.Metric`，html页面会正常显示全称链接。

    c. 如果html页面不想显示全称链接，只显示接口名，可写成 :class:`~.Metric`。

    注意：其中b,c两种写法，如果不同模块有同名接口，请多加上一些判断用模块，比如 :class:`.train.Metric`，:class:`~.train.Metric`。

10. rst文档内换行的用法

    a. rst文档没有一行内字数的限制，完整的一句话可以写在一行内；

    b. 如果需要换行书写，尽量在标点或连接特殊格式处换行。

    c. 上述两种情况都无法满足时，需在换行末尾加上换行符, 例如：

    ```text
    生成的配置文件内容示例如下，"remark"字段描述是否进行了数据处理参数调优，"summary"字段简要展示了数据处理流水线中\
    各个操作及其对应的最优配置，而"tree"字段则为完整的数据处理流水线结构信息。

    它由美国马萨诸塞理工学院的Gary B. Huang等人于2007年发布。该数据集包含13,233个人的近\
    50,000张图像，这些图像来自互联网上不同来源的人物照片，并包含了不同的姿势、光照

    .. note::
        - 当 `self` 和 `other` 的shape不同时，
          它们必须能够广播到一个共同的shape。
        - `self`、 `other` 和 `alpha` 遵守隐式类型转换规则以使数据类型\
          保持一致。

    参数：
        - **all_nodes** (bool) - 获取所有节点，包括在 `CallFunction` 节点、 `CellContainer` 节点和\
          子SymbolTree里面的节点。默认值： ``False`` 。
    ```

## 参考

- 有关rst的书写规则，请参考[rst入门](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)。
- 有关rst在Python领域的用法，请参考[rst在Python领域用法](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain)。
