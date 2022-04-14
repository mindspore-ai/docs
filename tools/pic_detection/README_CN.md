# 图片检查工具

## 简介

此工具可以检查用户指定目录里所有图片的使用情况，会检查出没有使用的图片，并且将没有使用的图片删除。

## 使用说明

该工具所依赖的操作系统为Windows操作系统，执行环境为Python环境，具体使用步骤如下所示：

1. 打开Git Bash，下载MindSpore Docs仓代码。

   ```bash
   git clone https://gitee.com/mindspore/docs.git -b r1.7
   ```

2. 进入`tools/pic_detection`目录。

   ```bash
   cd tools/pic_detection
   ```

3. 在`pic_detection`目录下执行如下命令，在输入需要检测目录的绝对路径后，开始进行检测，最后将没有使用的图片删除。

   ```bash
   python pic_detection.py
   ```

   > 检测目录的绝对路径全使用英文，并且使用Linux的绝对路径方式，例如：`/d/master/docs`。
