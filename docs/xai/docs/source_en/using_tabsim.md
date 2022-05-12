# Using TabSim Data Simulator

<a href="https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_en/using_tabsim.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Introduction

Sometimes, it is impossible to get sufficient amount of data for modeling or algorithm development, TabSim can be used for capturing the distribution of the tabular data and generating simulated data afterward.

The complete code of the tutorial below is [using_tabsim.py](https://gitee.com/mindspore/xai/blob/master/examples/using_tabsim.py).

## Installation

TabSim is part of the XAI package, no extra installation is required besides [MindSpore](https://mindspore.cn/install) and [XAI](https://www.mindspore.cn/xai/docs/en/master/installation.html).

## User Flow

There are 2 phases in the TabSim user flow:

1. Digestion: Analyzing the real tabular data, capturing the statistic characteristics and output the digest file. Accomplished by the commandline tool `mindspore_xai tabdig`.
2. Simulation: Generating simulated data according to the statistics stores in the digested file. Accomplished by the commandline tool `mindspore_xai tabsim`.

## Digestion Phase

```bash
mindspore_xai tabdig <real datafile> <digest file> [--bins <bins>] [--clip-sd <clip sd>]
```

`<real datafile>`: Path of the real CSV table to be simulated.

`<digest file>`: Path of the digest file to be saved.

`<bins>`: [optional] Number of bins for discretizing numeric columns, default: 10

`<clip sd>`: [optional] Number of standard deviations away from the mean that defines the outliers, outlier values
will be clipped. default: 3, setting to 0 or less will disable the value clipping.

### File Format of Real Data

The real data must be a CSV file with header that contains the names and type of all columns.

Header pattern: `<col name>|<col type>,<col name>|<col type>,<col name>|<col type>,...`

`<col name>`: Column name, allowed pattern: `[0-9a-zA-Z_\-]+`

`<col type>`: Column type, options: 'int', 'float', 'str', 'cat'

- 'int': Integers
- 'float': Float numbers
- 'str': Strings, allowed pattern: `[0-9a-zA-Z_\-\+\.]*`
- 'cat': Catergorical values, the underlying data type is integer without order

'int' and 'float' are numeric, while 'str' and 'cat' are discrete columns. There are at most 256 distinct values allowed
in each discrete column.

Optionally, users may specify at most one label column by adding a '*' before a discrete column (numeric columns are not
allowed).

Header example: `col_A|int,col_B|float,col_C|str,*col_D|cat`

It is recommended users randomly pick around 1 million records from the real database to form the real data file for the
statistic accuracy and the memory constraints.

### File Format of Digest Files

Digest files are clear text json, they stores the name, type and value distributions of columns without any actual record.
Users should never modify the digest file manually; otherwise that may corrupt it.

### Digestion Example

We used the [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) dataset for the
demonstration. This dataset consists of 3 different types of irises’ petal and sepal lengths. The Python code below
writes tabular data to `real_table.csv`.

```python
import sklearn.datasets

iris = sklearn.datasets.load_iris()
features = iris.data
labels = iris.target
# save the tabular data to file
header = 'sepal_length|float,sepal_width|float,petal_length|float,petal_width|float,*class|cat'
with open('real_table.csv', 'w') as f:
    f.write(header + '\n')
    for i in range(len(labels)):
        for feat in features[i]:
            f.write("{},".format(feat))
        f.write("{}\n".format(labels[i]))
```

Content of `real_table.csv`:

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

Then, we analyze the real tabular data, capturing the statistic characteristics and output it to `digest.json`.

```bash
mindspore_xai tabdig real_table.csv digest.json
```

Content of `digest.json`:

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

## Simulation Phase

```bash
mindspore_xai tabsim <digest file> <sim datafile> <rows> [--batch-size <batch size>] [--noise <noise>]
```

`<digest file>`: Path of the digest file of the real data.

`<sim datafile>`: Path of the simulated CSV table.

`<rows>`: Number of rows to be generated to `<sim datafile>`.

`<batch size>`: [optional] Number of rows in each batch, default: 10000

`<noise>`: [optional] Noise level (0.0-1.0) of value picking probabilities, 0.0 means 100% follows the digested joint
distributions, higher the noise level more even the probabilities. default: 0

### File Format of Simulated Data

The simulated CSV file has a similar format to the real data CSV file, but the header is different:

`<col name>,<col name>,<col name>,...`

It contains no `<col type>` and no `*` while the column order remains the same.

### Simulation Example

Here we generate 200000 rows of simulated data according to the statistics stored in the digested file, the simulated
data is output to `sim_table.csv`.

```bash
mindspore_xai tabsim digest.json sim_table.csv 200000
```

Content of `sim_table.csv`:

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
