# 导出py文件里的注释内容

## 简介

此工具可以遍历指定目录里所有的py文件，导出所有的注释内容，并用LanguageTool软件进行扫描，给出违反规则的txt格式的扫描结果，把txt格式转化为html格式，最终把所有的html整合成一个总的html页面，放入原来的目录下。

## 使用说明

该工具所依赖的操作系统为Linux操作系统，命令行执行的脚本，具体使用步骤如下所示：

1. 将输入文件或文件夹放入指定目录下，将export_annotation.py、commandline_extract_rulemessage.py、txttohtml.py、annotation_rulemessage_html.py也放进去。

   ```bash
   python annotation_rulemessage_html.py <your_file> <abspath(languagetool-commandline.jar)>
   ```

   - <your_file>为要检查的文件目录
   - <abspath(languagetool-commandline.jar)>为系统中languagetool-commandline.jar所在的绝对路径

   例如：`python annotation_rulemessage_html.py input /home/LanguageTool-5.4/languagetool-commandline.jar`(input为文件夹)。

2. 如果是检测目录内容，跑完上述代码后，会在原来目录下生成对应py文件的-annotation.txt为后缀的txt文件和-annotation-rulemessage.txt为后缀的txt文件和.html为后缀的html文件还有总的allrule.html。
