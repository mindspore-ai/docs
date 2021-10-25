# notebook检测

<!-- TOC -->

- [notebook检测](#notebook检测)
    - [检测原理](#检测原理)
    - [环境准备](#环境准备)
    - [检测配置](#检测配置)
    - [执行检测](#执行检测)
    - [检测结果](#检测结果)
    - [报错码](#报错码)

<!-- /TOC -->

## 检测原理

Notebook文档后缀为`.ipynb`。文档内容主要由python代码和markdown文档两部分组成，所以`notebook_lint.py`的检测原理是采用pylint工具和markdownlint工具分别对notebook文档的代码和markdown进行检测；其中python代码检测rule可以参考<https://gitee.com/mindspore/ms-pipeline/blob/master/pipeline/conf/rules/pylint/pylintrc>，markdownlint的检测的rule参考<https://gitee.com/mindspore/ms-pipeline/blob/master/pipeline/conf/rules/markdownlint/markdownlint_docs.rb>。

## 环境准备

根据检测原理，需要分别安装markdownlint工具和pylint工具。

- 安装markdownlint

    Markdownlint是用ruby编写的，并且作为一个rubygem发布，需要安装最新的ruby。

    安装[Ruby](http://rubyinstaller.org/downloads)。

    安装[Rubygems](https://rubygems/org/pages/download)

    完成上面两个依赖安装后，执行下述命令即可安装markdownlint。

    ```shell
    gem install mdl
    ```

- pylint的安装

    执行以下命令安装`pylint`。

    ```shell
    pip install pylint
    ```

## 检测配置

安装完成检测工具后，需要对检测工具进行配置，设置检测的忽略项及检测的严格程度。工具配置分别如下：

- markdownlint配置

    在命令路径上新建markdown配置文件[markdownlint_docs.rb](https://gitee.com/mindspore/ms-pipeline/blob/master/pipeline/conf/rules/markdownlint/markdownlint_docs.rb)。

    例如：`MD013`，`MD002`，`MD041`报错信息无需关注，则可以在`markdownlint_docs.rb`文件中添加如下三行内容：

    ```text
    exclude_rule 'MD013'
    exclude_rule 'MD002'
    exclude_rule 'MD041'
    ```

- pylint配置

    在命令路径上新建pylint配置文件[.pylintrc](https://gitee.com/mindspore/ms-pipeline/blob/master/pipeline/conf/rules/pylint/pylintrc)

    例如：`C0413`,`C0412`,`C0411`报错信息无需关注，则可以在`.pylintrc`文件中的`disable`进行如下设置：

    ```text
    disable=C0413,C0412,C0411
    ```

- 检测脚本`notebook_lint.py`

    从docs仓库下载检测脚本`notebook_lint.py`放置在检测路径中。

文档的目录结构如下：

```text
/usr1/check_tool/
|___notebook_lint.py
|___markdownlint_docs.rb
|___.pylintrc
```

Markdownlint是用ruby编写的，并且作为一个rubygem发布，只要您的系统有相对最新的ruby，Markdownlint的安装和使用就很简单。有两种方式进行安装

## 执行检测

执行检测可以对单个文件进行检测，也可以对目录进行检测

如果需要对`xxx1.ipynb`和`xxxx2.ipynb`进行检测，可以输入如下命令：

```bash
/user1/check_tool# python notebook_lint.py {xxxx1.ipynb的路径} {xxxx2.ipynb的路径}
```

如果需要对`docs`仓的所有notebook进行检测，可以输入如下命令：

```text
/user1/check_tool# python notebook_lint.py {docs仓的路径}
```

## 检测结果

检测结果的打印信息经过了部分调整。报错信息的组成如下：

```text
{检测文件}：{报错的Cell}：{该报错Cell中第几行}：{报错信息}"{报错行的内容}"
```

> 其中报错行的内容长度大于30时，多出的内容用`...`省略。

详细报错内容举例如下：

```text
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_22:139:4: W0221: Parameters differ from overridden 'construct' method (arguments-differ) "    def construct(self, inputs..."
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_24:4:0: W0404: Reimport 'nn' (imported line 290) (reimported) "from mindspore import nn
"
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_4:11:MD009 Trailing spaces "| Review  | Label  |
"
```

以报错信息的第一行内容举例：

`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_22:139:4: W0221: Parameters differ from overridden 'construct' method (arguments-differ) "    def construct(self, inputs..."`

- 检测文件：docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb

- 报错的Cell：cell_22。即检测文件中第22个cell报错。

- 报错的Cell中第几行：139。即第22个cell中第139行报错。

- 报错信息：4: W0221: Parameters differ from overridden 'construct' method (arguments-differ)。即报错码为W0221对应的检测规则码。

- 报错行的内容： "    def construct(self, inputs..."。即该报错行的具体内容。

## 检测规则

### markdown的报错码

报错码以`MD`开头：

可参考链接：<https://github.com/DavidAnson/markdownlint>

### pylint的检测规则

可参考链接：<https://tools.mindspore.cn/tools/check/pylint/rules/pylintrc>
