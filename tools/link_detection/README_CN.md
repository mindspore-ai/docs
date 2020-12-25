# 链接检查工具

## 简介

此工具可以检查用户指定目录里所有文件的链接，将所有链接分为三类，并且将检查结果分别写入三个文件，如下所示：

1. 响应的状态码不是200的链接，写入`400.txt`文件中。
2. 脚本执行过程中请求出现异常的链接，写入`exception.txt`文件中。
3. 对于安装包的链接，因为请求非常耗时，所以不发请求，直接写入`slow.txt`文件中。

## 使用说明

该工具所依赖的操作系统为Windows操作系统，执行环境为Python环境，具体使用步骤如下所示：

1. 打开Git Bash，下载MindSpore Docs仓代码。

   ```shell
   git clone https://gitee.com/mindspore/docs.git -b r1.1
   ```

2. 进入`tools/link_detection`目录，安装执行所需的第三方库。

   ```shell
   cd tools/link_detection
   pip install requests
   ```

3. 在`link_detection`目录下执行如下命令，在输入需要检测目录的绝对路径后，开始进行检测，完成后会在当前目录下新建`404.txt`、`exception.txt`、`slow.txt`三个文件。

   ```shell
   python link_detection.py
   ```

   > 检测目录的绝对路径全使用英文，并且使用Linux的绝对路径方式，例如：`/d/master/docs`。
