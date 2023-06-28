# 贡献文档

[View English](./CONTRIBUTING_DOC.md)

欢迎参与MindSpore文档贡献，符合要求的文档将会在[MindSpore官网](http://www.mindspore.cn)中呈现。

## 新建或更新文档

本项目支持MarkDown和reStructuredText格式的内容贡献，对应地可创建```.md```和```.rst```为后缀的文档或修改已存在的文档。

## 提交修改

提交修改的步骤与代码相同，请参考[代码贡献指南](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 文档写作规范

- 标题仅支持Atx风格，标题与上下文需用空行隔开。

  ```markdown
  # 一级标题

  ## 二级标题

  ### 三级标题
  ```

- 列表标题和内容如需换行显示，标题和内容间需增加一个空行，否则无法实现换行。

  ```markdown
  - 标题

    内容。
  ```

- 目录中的锚点（超链接）只能使用中文、小写英文字母或“-”，不能带有空格或其他特殊字符，否则会导致链接无效。

- 注意事项使用“>”标识。

  ```markdown
  > 注意事项内容。
  ```  

- 参考文献需列举在文末，并在文中标注。

  ```markdown
  引用文字或图片说明后，增加标注[编号]。

  ## 参考文献

  [1] 作者. [有链接的文献名](http://xxx).

  [2] 作者. 没有链接的文献名.
  ```

- 示例代码注释需遵循如下要求：

    - 注释用英文写作；
    - Python函数、方法、类的注释使用```"""```；
    - Python其他代码注释使用```#```；
    - C++代码注释使用```//```。

  ```markdown
  """
  Python函数、方法、类的注释
  """

  # Python代码注释

  // C++代码注释

  ```

- 图和图标题前后需增加一个空行，否则会导致排版异常。正确举例如下：

   ```markdown
  如下图所示：

  ![](./xxx.png)

  图1：xxx

  下文内容。
  ```

- 表格前后需增加一个空行，否则会导致排版异常。有序或无序列表内不支持表格。正确举例如下：

  ```markdown
  ## 文章标题

  | 表头1   | 表头2
  | :-----  | :----
  | 内容I1  | 内容I2
  | 内容II1 | 内容II2

  下文内容。
  ```

- 教程、文档中引用接口、路径名、文件名等使用“\` \`”标注，如果是函数或方法，最后不加括号。举例如下：

    - 引用方法

    ```markdown
    使用映射 `map` 方法。
    ```

    - 引用代码

    ```markdown
    `batch_size`：每组包含的数据个数。
    ```

    - 引用路径

    ```markdown
    将数据集解压存放到工作区`./MNIST_Data`路径下。
    ```

    - 引用文件名

    ```markdown
    其他依赖项在`requirements.txt`中有详细描述。
    ```

- 教程、文档中待用户替换的内容需要额外标注，在正文中，使用“*”包围需要替换内容，在代码片段中，使用“{}”包围替换内容。举例如下：

    - 正文中

    ```markdown
    需要替换你的本地路径*your_path*。
    ```

    - 代码片段中

    ```markdown
    conda activate {your_env_name}
    ```

## 文档检查

Markdownlint是一款检查Markdown文件格式正确性的工具，可以根据设置的规则以及创建的新规则对Markdown文件进行全面的检查。

其中，MindSpore CI 在默认配置的基础上，修改了如下规则：

MD007（无序列表缩进）规则将参数indent设置为4；MD009（行尾空格）规则将参数br_spaces设置为2；MD029（有序列表的前缀序号）规则将参数style设置为ordered。

详细规则信息请参考[RULES](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md)。
