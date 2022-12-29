# 中文API格式规范

## 类 Class

```text
.. py:class:: Tensor(default_input, dtype=None, init_with_data=True)

    Tensor简介。

    参数：
        - **参数1** (Tensor) – 参数1说明。
        - **参数2** (int, 可选) – 参数2说明。默认值：None。

          - **二级参数1** (int) – 二级参数1说明。（注意：二级参数需要和上面一级参数的“*”对齐。）
          - **二级参数2** (int) – 二级参数2说明。

        - **参数3** (bool, 可选) – 参数3说明。默认值：True。

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
        - **Error1** – 异常描述2。
```

## 属性 Property

```text
.. py:method:: T
    :property:

    返回转置后的张量。
```

注意：相对于类要增加4空格缩进。

## 方法 Method

```text
.. py:method:: abs()

    方法说明。

    返回：
        返回说明。
```

注意：相对于类要增加4空格的缩进。

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

## 函数 Function

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
        - **Error1** – 异常描述2。
```

## Note

```text
.. note::
    此处描述具体需要注意的部分，前面需缩进四格。
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

## 内容注意事项

1. 类（class，如mindspore.nn模块）文档中包含参数、输入、输出、异常；函数（function，如mindspore.ops模块）和方法（method，如mindspore.Tensor中的方法）文档中包含参数、返回、异常。

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

    请参考 `tensor <https://www.gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/tensor.py>`_ 。
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
8. `mindspore.train`和`mindspore.nn.transformer`模块下的接口，需要将每个API的文档分别放于`train`和`transformer`目录下，并在`mindspore.train.rst`和`mindspore.nn.transformer.rst`文件中通过`.. include::`的写法将接口引入过来；其他模块下的接口结构参考英文的API文档。

9. 引用类/函数的用法

    ```text

    :class:`类的全称`

    :func:`函数的全称`

    ```

    a. 引用其他类的内容，类的全称类似于mindspore.train.Metric，包含一级二级类别的名称。引用其他函数的内容，函数的全称类似于mindspore.ops.dropout。

    b. 如果简写为Metric，英文书写为 :class:`Metric`，中文书写为 :class:`.Metric`，简写中文前面需要加.。

## 参考

- 有关rst的书写规则，请参考[rst入门](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)。
- 有关rst在Python领域的用法，请参考[rst在Python领域用法](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain)。
