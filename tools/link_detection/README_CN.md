# 链接检测

## 环境准备

安装第三方依赖

```bash
pip install requests
```

## 执行检测

执行检测可以对单个文件进行检测，也可以对目录进行检测

```bash
/user1/check_tool# python link_lint.py -p=file1,file2,dir1,dir2
```

## 检测结果

检测结果的打印信息经过了部分调整。报错信息的组成如下：

```text
{检测文件}：{报错内容}
```

详细报错内容举例如下：

```text
ERROR:docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb: line_2: Error link: https://xxxxx
```

以报错信息内容举例：

`ERROR:docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb: line_2: Error link: https://xxxxx`

- 链接检测状态：`ERROR`表示检测出了链接报错，如果是`WARRING`表示该失效链接属于`gitee.com`中的链接，可能由于文件正在合入而造成的链接报错。

- 检测文件：`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb`

- 报错行：`line_2`，表示报错链接位于文档中的第二行。

- 报错的链接: `Error link: https://xxxxx`。即具体的报错链接。

## 检测白名单设置

`filter_linklint.txt`文件中存储着链接检测的白名单，每行可写一个链接列入白名单。
该文件默认放在与`link_lint.py`同目录。也可以通过命令传入`--white-list={白名单文件地址}`载入指定地址的白名单文件。

```bash
python link_lint.py --white-list xxx/xxx/xx.txt
```

白名单内容可书写如下：

```text
https://xxxxx.com
https://xxx.com/xxx.html
...
```

注意：

- 每行只能写一条白名单链接。
- 支持使用`*`来进行白名单链接的模糊匹配。例如：`https://www.mindspore*/tutorials*`