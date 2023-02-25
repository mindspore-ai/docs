# 规则

- [规则](#规则)
    - [Title underline too short](#title-underline-too-short)
    - [Inline strong start-string without end-string](#inline-strong-start-string-without-end-string)
    - [Bullet list ends without a blank line](#bullet-list-ends-without-a-blank-line)
    - [unexpected unindent](#unexpected-unindent)
    - [Error in "xxx" directive](#error-in-"xxx"-directive)
    - [Inline interpreted text or phrase reference start-string without end-string](#inline-interpreted-text-or-phrase-reference-start-string-without-end-string)
    - [Content block expected for the "xxx" directive; none found](#content-block-expected-for-the-"xxx"-directive-none-found)
    - [There are mismatched or missing "xxx" in the statements](#there-are-mismatched-or-missing-"xxx"-in-the-statements)

## Title underline too short

表示标题下的 `=`符号长度不够，需要等于标题长度，若标题为中文，则需要大于标题长度。

- 正确示例

    ```text
    mindspore.nn.Cell
    ====================
    ```

- 错误示例

    ```text
    mindspore.nn.Cell
    ============
    ```

## Inline strong start-string without end-string

表示强调符号 `*`左右或缺失。

- 正确示例

    ```text
    **支持平台**
    ```

- 错误示例

    ```text
    **支持平台*：
    ```

## Bullet list ends without a blank line

表示无序列表整体上下缺少空白行，或者存在换行时缩进没正确对齐。

- 正确示例1：

    ```text
    描述如下：

    - 第一个描述。
    - 第二个描述。

    后续内容
    ```

- 错误示例1：

    ```text
    描述如下：
    - 第一个描述。
    - 第二个描述。
    后续内容
    ```

- 正确示例2：

    ```text
    描述如下：

    - 第一个长描述内容及
      换行。
    - 第二个描述内容。
    ```

- 错误示例2：

    ```text
    描述如下：

    - 第一个长描述内容及
    换行
    - 第二个描述内容
    ```

## unexpected unindent

表示缩进位置不对。

- 正确示例：

    ```text
    - 第一个描述。
    - 第二个描述。
    ```

- 错误示例：

    ```text
    - 第一个描述
     - 第二个描述
    ```

## Error in "xxx" directive

表示所在位置命令格式写法错误。

- 正确示例：

    ```text
    .. py:class:: mindspore.nn.Cell
    ```

- 错误示例：

    ```text
    . py:class:: mindspore.nn.Cell
    ```

## Inline interpreted text or phrase reference start-string without end-string

表示块符号\`左右或缺失。

- 正确示例：

    ```text
    需要强调的是`var`变量
    ```

- 错误示例：

    ```text
    需要强调的是`var变量
    ```

## Content block expected for the "xxx" directive; none found

表示所在位置命令下面缺少空行，或者所在位置命令下没有具体内容。

- 正确示例1：

    ```text
    .. code-block::

        import mindspore
        import sys
    ```

- 错误示例1：

    ```text
    .. code-block::
        import mindspore
        import sys
    ```

- 正确示例2：

    ```text
    .. code-block::

        import mindspore
        import sys
    ```

- 错误示例2：

    ```text
    .. code-block::

    import mindspore
    import sys
    ```

## There are mismatched or missing "xxx" in the statements

表示所在位置的某个括号缺失对应匹配括号或者匹配的括号错误。(描述变量的范围时请使用英文逗号，否则也会报错)

- 正确示例1：

    ```text
    输入：
        - **shape** (Union[Tuple[int], Tensor[int]]) - 1-D Tensor或Tuple，指定了输出Tensor的shape。
          其数据类型必须是int32或int64。
    ```

- 错误示例1：

    ```text
    输入：
        - **shape** (Union[Tuple[int], Tensor[int]) - 1-D Tensor或Tuple，指定了输出Tensor的shape。
          其数据类型必须是int32或int64。
    ```

- 正确示例2：

    ```text
    如果算子Log的输入值在(0, 0.01)或[0.95, 1.05]范围内，则输出精度可能会存在误差。
    ```

- 错误示例2：

    ```text
    如果算子Log的输入值在(0, 0.01）或[0.95, 1.05]范围内，则输出精度可能会存在误差。
    ```