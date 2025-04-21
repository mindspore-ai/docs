# 可视化工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/visual_tool.md)

## 概述

[Netron](https://github.com/lutzroeder/netron)是一个基于[Electron](http://www.electronjs.org/)平台开发的神经网络模型可视化工具，支持许多主流AI框架模型的可视化，支持多种平台（Mac、Windows、Linux等）。`Netron`支持MindSpore Lite模型，可以方便地查看模型信息。如下图所示，使用`Netron`加载`.ms`模型后，可以展示模型的拓扑结构和图、节点的信息等。

![img](../images/visual_mnist.png)

## 功能列表

- 支持加载`.ms`模型，要求MindSpore版本>=1.2.0;
- 支持查看子图；
- 支持拓扑结构和数据流`shape`的展示；
- 支持查看模型的`format`、`input`和`output`等；
- 支持查看节点的`type`、`name`、`attribute`、`input`和`output`等；
- 支持结构化的`weight`、`bias`等数据的查看与保存;
- 支持可视化结果导出为图片保存。

## 使用方式

ms模型的支持代码已经合入官方库。`Netron`的下载地址为 <https://github.com/lutzroeder/netron/releases/latest>，作者不定期更新并发布Release版本。用户按照以下方式安装`Netron`，将模型拖入窗口即可打开。

- macOS: 下载`.dmg`文件或者执行`brew cask install netron`

- Linux: 下载`.AppImage`文件或者执行`snap install netron`

- Windows: 下载`.exe`文件或者执行`winget install netron`

- Python服务器：执行`pip install netron`安装Netron，然后通过`netron [FILE]`或`netron.start('[FILE]')`加载模型

- 浏览器：打开 <https://netron.app/>

## 开发调试

### 使用开发版本

步骤1：通过`git clone https://github.com/lutzroeder/netron`克隆一份源码

步骤2：进入`netron`目录，执行`npm install`

步骤3：执行`make build`进行编译，在./dist路径下将生成可执行程序

### 使用Javacript调试模型

在调试模型时，在`netron`文件夹下，先在`./test/models.json`中添加调试模型的信息，然后使用`node.js`调试`./test/model.js`脚本即可。
