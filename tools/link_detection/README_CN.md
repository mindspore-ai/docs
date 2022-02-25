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
docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb: 404: Error links in the line
```

以报错信息内容举例：

`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb: 404: Error link in the file.{失效链接}`

- 检测文件：`docs/tutorials/source_zh_cn/intermediate/text/sentimentnet.ipynb`

- 报错代码：`404`。即该行中存在状态码是404的链接，即不存在该网址。

- 报错的链接。

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