# rst文件格式检查工具

## 简介

此工具用于检测`.rst`文档内的格式问题。

## 工具使用说明

在命令行中执行`ms_rst_lint.py`：

`python ms_rst_lint.py --level {debug,info,warning,error,severe} {file_path}`

其中：

- `file_path`为 `.rst`文件或目录路径名。
- `--level`为输出信息报错级别（debug、info、warning、error、severe），可省略，默认为 `warning`。

输出报错信息如下所示：

`{level} {rst_file_path}:{line_number} {error_info}`

其中：

- `level`为错误级别，`WARNING`或者 `ERROR`；
- `rst_file_path`为检测文档路径名；
- `line_number`为错误所在行;
- `error_info`为详细[错误信息](https://gitee.com/mindspore/docs/blob/r1.7/tools/rst_lint/RULES.md#)。

## 示例

`test.rst`文件内容如下所示：

```rst
mindspore.nn.Cell
===============

.. py:class:: mindspore.nn.Cell(auto_prefix=True, flags=None)

     所有神经网络的基类。

   一个 `Cell` 可以是一个单一的神经网络单元，如conv2d， relu， batch_norm等，也可以是组成网络的 `Cell` 的结合体。

   .. note:: 一般情况下，自动微分 (AutoDiff) 算法会自动生成梯度函数的实现，但是如果实现了自定义的反向传播函数 (bprop method)，梯度函数将会被Cell中定义的反向传播函数代替。反向传播函数将会接收一个包含关于输出的损失梯度的张量 `dout` 和一个包含前向传播结果的张量 `out` 。反向传播函数需要计算关于输入的损失梯度，关于参数变量的损失函数目前还不支持。反向传播函数必须包含self参数。

   **参数* ：

      - **auto_prefix** (`Cell`) – 递归生成命名空间。默认值：True。
       - **flags** (`dict`) - 网络配置信息，目前用于网络和数据集的绑定。用户还可以通过该参数自定义网络属性。默认值：None。

   **支持平台**：

   ``Ascend`` ``GPU`` ``CPU``

   **样例** :

      .. code-block::
            >>> import mindspore.nn as nn
            >>> import mindspore.ops as ops
            >>> class MyCell(nn.Cell):
            ...    def __init__(self):
            ...        super(MyCell, self).__init__()
            ...        self.relu = ops.ReLU()
            ...
            ...    def construct(self, x):
            ...        return self.relu(x)
   .. py:property:: bprop_debug

      获取单元格自定义反向传播调试是否已启用。

   .. py:method:: cast_param(param)
      在pynative模式下，根据自动混合精度的级别转换权重类型。
```

在 `test.rst`所在目录执行以下命令：

`python ms_rst_lint.py test.rst`

输出结果如下所示：

```text
WARNING test.rst:2 Title underline too short.
WARNING test.rst:12 Inline strong start-string without end-string.
WARNING test.rst:15 Bullet list ends without a blank line; unexpected unindent.
ERROR test.rst:23 Error in "code-block" directive: maximum 1 argument(s) allowed, 31 supplied.
WARNING test.rst:33 Block quote ends without a blank line; unexpected unindent.
ERROR test.rst:33 Content block expected for the "py:property" directive; none found.
ERROR test.rst:36 Content block expected for the "py:method" directive; none found.
```

根据输出显示，在2，12，15，23，33，36等行位置出现有相应格式错误。
