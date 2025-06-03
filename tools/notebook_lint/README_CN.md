# notebook检测

## notebook_lint.py检测原理简介

`notebook_lint.py`主要以文件5个类对象组成，分别为:`ReadFile`、`CustomCheck`、`Notebook_Pylint`和`Notebook_Markdownlint`,`PrintInfo`，其功能分别如下：

- `ReadFile`：传入文件路径对象，将其初始化为`json`格式对象，以`cell`为元素的list列表，以`markdown`类型的cell组成的列表，以`code`类型的cell组成的列表。
- `CustomCheck`：自定义检测规则，对内容进行自定义的检测规则构建，构建完成后传入`check`方法中对文档内容执行检测。
- `Notebook_Pylint`：以`pylint`作为检测工具的对象，其执行检测的方法为`check`。
- `Notebook_Markdownlint`: 以`markdownlint`作为检测工具的对象，其执行检测的方法为`check`。
- `PrintInfo`：传入各个检测对象的检测结果，并将检测信息过滤后打印出来。

> 检测结果的输出值为`list`格式，其中元素值格式为`(文件名, 报错单元, 报错单元行, 报错码, 报错信息)`

## 环境准备

- 安装markdownlint

    Markdownlint是用ruby编写的，并且作为一个rubygem发布，需要安装最新的ruby。

    安装[Ruby](http://rubyinstaller.org/downloads)。

    安装[Rubygems](https://rubygems.org/pages/download)

    完成上面两个依赖安装后，执行下述命令即可安装markdownlint。

    ```shell
    gem install mdl
    ```

- pylint的安装

    执行以下命令安装`pylint`。

    ```shell
    pip install pylint
    ```

## 执行检测

执行检测可以对单个文件进行检测，也可以对目录进行检测

- 对文件检测

    如果需要对`xxx1.ipynb`和`xxxx2.ipynb`进行检测，可以输入如下命令：

    ```bash
    /user1/check_tool# python notebook_lint.py {xxxx1.ipynb的路径} {xxxx2.ipynb的路径}
    ```

- 对目录进行检测

    如果需要对`docs`仓的所有notebook进行检测，可以输入如下命令：

    ```text
    /user1/check_tool# python notebook_lint.py {docs仓的路径}
    ```

- 忽略检测项

    如需忽略部分检测项目，可以使用`--ignore=<error code>`进行忽略，例如：

    ```bash
    /user1/check_tool# python notebook_lint.py --ignore=MD029,R0901,... {xxxx1.ipynb的路径}
    ```

## 检测结果

检测结果的打印信息经过了部分调整。报错信息的组成如下：

```text
{检测文件}{报错的Cell}{该报错Cell中第几行}{报错信息}
```

详细报错内容举例如下：

```text
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_22:139:4: W0221: Parameters differ from overridden 'construct' method (arguments-differ)
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_24:4:0: W0404: Reimport 'nn' (imported line 290) (reimported)
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_4:11:MD009 Trailing spaces
```

以报错信息的第一行内容举例：

`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb:cell_22:139:4: W0221: Parameters differ from overridden 'construct' method (arguments-differ)

- 检测文件：`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb`

- 报错的Cell：`cell_22`。即检测文件中第22个cell报错。

- 报错的Cell中第几行：139。即第22个cell中第139行报错。

- 报错信息：`4: W0221: Parameters differ from overridden 'construct' method (arguments-differ)`。即报错码为W0221对应的检测规则码。

## 检测规则

### 自定义检测规则报错

报错码以`CC`开头：

MD099: 数学公式与文档内容之间应有空行隔开。

错误举例一：

```text
表示为一个关于输入$x$的函数$f(x)$，其中$x$为$0$到$N-1$之间的整数。那么，函数$f$定义为：
$$
\begin{equation}
f(x)=\begin{cases}0,x\neq x_{target}\\\\
1,x=x_{target}
\end{cases}
\end{equation}.
$$

其中f(x)表示...
```

错误举例二：

```text
表示为一个关于输入$x$的函数$f(x)$，其中$x$为$0$到$N-1$之间的整数。那么，函数$f$定义为：

$$
\begin{equation}
f(x)=\begin{cases}0,x\neq x_{target}\\\\
1,x=x_{target}
\end{cases}
\end{equation}.
$$
其中f(x)表示...
```

错误举例三：

```text
表示为一个关于输入$x$的函数$f(x)$，其中$x$为$0$到$N-1$之间的整数。那么，函数$f$定义为：$$\begin{equation}
f(x)=\begin{cases}0,x\neq x_{target}\\\\
1,x=x_{target}
\end{cases}
\end{equation}.$$
其中f(x)表示...
```

正确示例：

```text
表示为一个关于输入$x$的函数$f(x)$，其中$x$为$0$到$N-1$之间的整数。那么，函数$f$定义为：

$$
\begin{equation}
f(x)=\begin{cases}0,x\neq x_{target}\\\\
1,x=x_{target}
\end{cases}
\end{equation}.
$$

其中f(x)表示...
```

### markdown的报错码

报错码以`MD`开头：

可参考链接：<https://github.com/DavidAnson/markdownlint>

### pylint的检测规则

可参考链接：<https://pylint.pycqa.org/en/latest/technical_reference/features.html>
