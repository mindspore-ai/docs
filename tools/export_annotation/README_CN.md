# 导出py文件里的注释内容

## 简介

此工具可以导出py文件内的所有注释内容或者遍历文件夹里所有py文件并导出所有的注释内容，并且结果写入特定的文件夹中，建立与输入文件同等的目录，放入对应目录下。
此工具可作为languagetool工具检查注释语法的文件准备操作。

## 使用说明

该工具所依赖的操作系统为Linux操作系统，命令行执行的脚本，具体使用步骤如下所示：

1. 将输入文件或文件夹放入指定目录下，将export_filepy_annotation.py也放进去

   ```bash
   python export_filepy_annotation.py <your_file>
   ```

   - <your_file>为要检查的文件目录或文件

   例如：`python export_filepy_annotation.py array_ops.py`或者`python export_filepy_annotation.py input`(input为文件夹)。

2. 如果是检测目录内容，跑完上述代码后，会在同级目录下生成annotation的目录，打开后，是和输入目录同名的目录，里面包含了所有py文件中的注释内容放在了txt文档里。
