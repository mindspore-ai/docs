# 使用 TabSim 数据模拟器

<a href="https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_zh_cn/using_tabsim.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 简介

有些时候，工程师只能获取少量的数据，这些数据的数量不足以支持建模或者算法开发。TabSim 可以统计表格数据的特征分布，并根据这些信息生成大量的模拟数据。

以下教程的完整代码：[using_tabsim.py](https://gitee.com/mindspore/xai/blob/master/examples/using_tabsim.py)。

## 安装

TabSim 是 XAI 的一部份，用户在安装好 [MindSpore](https://mindspore.cn/install) 及 [XAI](https://www.mindspore.cn/xai/docs/zh-CN/master/installation.html) 后即可使用。

## 用户使用流程

TabSim 的用户流程分为两个阶段：

1. 消化阶段：分析真实的表格数据，统计数据特征分布并输出摘要文件，通过命令行工具 `mindspore_xai tabdig` 完成。
2. 模拟阶段：根据摘要文件中存储的统计信息生成模拟数据，通过命令行工具 `mindspore_xai tabsim` 完成。

## 消化阶段

```bash
mindspore_xai tabdig <real datafile> <digest file> [--bins <bins>] [--clip-sd <clip sd>]
```

`<real datafile>`：需要模拟的真实数据表格（.csv）的路径。

`<digest file>`：输出的摘要文件的保存路径。

`<bins>`：[可选] 离散数字列的箱数，默认值：10。

`<clip sd>`：[可选] 定义离群值的平均值偏离标准差数，离群值将被剪裁。 默认值：3，设置为 0 或以下将不会剪裁离群值。

### 真实数据格式

真实数据必须是csv格式的文件，它的标题（第一行）包含所有列的名称和类型。

标题模式：`<col name>|<col type>,<col name>|<col type>,<col name>|<col type>,...`

`<col name>`：列名，允许的模式： `[0-9a-zA-Z_\-]+`

`<col type>`：列类型，选项：'int', 'float', 'str', 'cat'

- 'int'：整数
- 'float'：浮点数
- 'str'：字符串，允许的正则表达式：`[0-9a-zA-Z_\-\+\.]*`
- 'cat'：分类值，底层数据类型为无序整数

'int' 和 'float' 是数字，而 'str' 和 'cat' 是离散列，每个离散列中最多允许 256 个不同的值。

或者，用户可以在离散列之前添加 '*' 来指定最多一个标签列（不允许使用数字列）。

标题示例: `col_A|int,col_B|float,col_C|str,*col_D|cat`

建议用户从真实数据库中随机抽取约一百万条记录以组成真实数据文件，这样可以保证统计的准确性，以及不会超出内存限制。

### 摘要文件格式

摘要文件为明文 json 文件，它没有储存任何真实数据，只有的列名称、类型和值分布。用户不应该手动修改摘要文件，否则可能会损坏它。

### 消化示例

我们使用 [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) 数据集进行演示，
这个数据集包含了三种鸢尾花的花瓣长度和萼片长度。下面的 Python 代码将这个数据集写入 `real_table.csv`。

```python
import sklearn.datasets

iris = sklearn.datasets.load_iris()
features = iris.data
labels = iris.target
# 将表格数据保存到文件
header = 'sepal_length|float,sepal_width|float,petal_length|float,petal_width|float,*class|cat'
with open('real_table.csv', 'w') as f:
    f.write(header + '\n')
    for i in range(len(labels)):
        for feat in features[i]:
            f.write("{},".format(feat))
        f.write("{}\n".format(labels[i]))
```

`real_table.csv` 的内容：

```text
sepal_length|float,sepal_width|float,petal_length|float,petal_width|float,*class|cat
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
...
```

然后，我们分析真实的表格数据，统计特征分布并输出摘要文件 `digest.json`。

```bash
mindspore_xai tabdig real_table.csv digest.json
```

`digest.json` 的内容：

```json
{
    "label_col_idx": 4,
    "columns": [
        {
            "name": "sepal_length",
            "idx": 0,
            "ctype": "float",
            "dtype": "float",
            "is_numeric": true,
            "is_label": false,
            ...
```

## 模拟阶段

```bash
mindspore_xai tabsim <digest file> <sim datafile> <rows> [--batch-size <batch size>] [--noise <noise>]
```

`<digest file>`：真实数据的摘要文件的路径。

`<sim datafile>`：输出的模拟数据的保存路径（.csv）。

`<rows>`：生成多少行模拟数据。

`<batch size>`：[可选] 每个批次的行数，默认值：10000。

`<noise>`：[可选] 0.0-1.0 取值概率的噪声级，0.0 表示完全遵循特征统计的联合分布，噪声级越高，概率越均匀。 默认值：0。

### 模拟数据格式

模拟数据文件与真实数据文件的格式类似，但标题不同：

`<col name>,<col name>,<col name>,...`

它不包含 `<col type>`，也不包含 '*' 。所有列的顺序保持不变。

### 模拟示例

这里我们根据摘要文件中储存的统计信息生成 200000 行模拟数据并储存到 `sim_table.csv` 。

```bash
mindspore_xai tabsim digest.json sim_table.csv 200000
```

`sim_table.csv` 的内容:

```text
sepal_length,sepal_width,petal_length,petal_width,class
5.577184113278916,2.600922272560204,4.432243573999988,1.3937476921377445,1
6.723739024436704,2.7995789972671985,4.093195099230183,1.377081159510022,1
4.787110003892638,2.8994714750972608,1.221068662892122,0.18023497892950327,0
5.47601589088659,2.683719381022501,4.429520567795243,1.44376166769605,1
5.713634033969561,2.238437659593092,4.468051986603512,1.5218876291352155,1
6.014412107785783,2.921972441210267,4.066770696930024,0.9183809029577147,1
6.188386742135447,2.92122446931648,5.288862927543273,1.4537708701756062,2
7.394485586937094,2.867479423550221,5.730391070749579,1.998759192383688,2
5.468839597899383,2.8957462954323083,4.4090170094158525,1.502682955942951,1
...
```
