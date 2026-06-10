<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} 已废弃（Deprecated）
:class: warning

本页属于 **静态图（GRAPH_MODE）实现** 章节，已标记为废弃。新特性优先在「r2.0.0 动态图实现」章节演进，请优先查阅动态图相关文档。
```

# MindSpore Transformers对接通用评测工具的实践案例

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/static_graph/example/model_test/model_test.md)

本文由Killjoy, chen-xialei, fuyao-15989607593, laozhuang, oacjiewen贡献。

在实际进行大模型开发时，用户基于MindSpore Transformers对模型训练/微调后，往往会使用通用评测工具在自定义数据集上进行模型能力的评测。本文介绍了MindSpore Transformers的模型通过部署后对接通用评测工具的实践，涵盖使用vLLM-MindSpore部署模型，并基于两种通用评测框架`lm-eval`和`opencompass`进行能力评测。通过本实践案例，您可以了解如何使用通用评测工具，对基于MindSpore Transformers训练/微调出的模型进行评测。

## 1. 环境准备

首先需安装以下环境以进行模型的部署和评测工作。

| 依赖软件           | 使用版本 |
|--------------------|----------|
| MindSpore Transformers |    1.6.0    |
| vLLM-MindSpore     | 0.5.0       |
| lm-eval            | 0.4.9       |
| opencompass              | 0.5.0      |

### 1.1 MindSpore Transformers

参考[MindSpore Transformers 环境安装](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)搭建环境。

### 1.2 vLLM-MindSpore

用户可执行以下命令，拉取vLLM-MindSpore插件代码仓库，并构建镜像：

```bash
git clone https://atomgit.com/mindspore/vllm-mindspore.git
bash build_image.sh
```

> 如果构造镜像遇到超时问题，可以尝试在`build_image.sh`脚本中添加 `ENV UV_HTTP_TIMEOUT=3000`，并且在仓库`install_depend_pkgs.sh`脚本中更换速度更快的镜像站。

用户根据服务器配置创建容器，详细请参考[容器创建教程](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/getting_started/installation/installation.html)。

### 1.3 lm-eval

**注意：极其推荐单开一个conda环境，python≥3.10，以避免某些兼容问题**

需要注意需要采用本地安装的方法，不要直接`pip install lm-eval`。

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

如果遇到报错`Error: Please make sure the libxml2 and libxslt development packages are installed`
使用如下命令进行安装：

```bash
conda install -c conda-forge libxml2 libxslt
```

另外为避免可能的版本兼容问题，需要把`datasets`库和`transformers`库安装为特定版本。

```bash
pip install datasets==2.18.0
pip install transformers==4.35.2
```

### 1.4 opencompass

```bash
pip install -U opencompass
```

在安装过程中，可能会遇到如下报错：

```bash
AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
```

解决方法：
改为由[源码](https://github.com/open-compass/opencompass)安装，将`requirements/runtime.txt`中的`pyext`和`rouge`删掉。

## 2. 部署模型

设置环境变量：

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers
```

部署模型：

```bash
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model MODEL_PATH --port YOUR_PORT --host 0.0.0.0 --served-model-name YOUR_MODEL_NAME
```

## 3. 使用lm-eval进行评测

lm-eval是一个大型综合评测框架，适用于众多通用领域测试集（MMLU、CEval等），同时支持方便的自定义数据测试。

### 3.1 处理数据集

> 该步为自定义数据集所需步骤，测试通用测试集时，直接使用[官方教程](https://github.com/EleutherAI/lm-evaluation-harness)即可。

假设本地存在一个用户自定义的数据集，该数据集为一个`csv`文件，每一条数据集是单选题，包含一个问题、四个选项、答案，共6个属性，该文件没有表头，文件名为 `output_filtered.csv`。首先执行如下代码，将`csv`文件转换为适合模型处理的`Dataset`格式：

```python
import pandas as pd
from datasets import Dataset, DatasetDict
import os

def convert_csv_to_parquet_dataset(csv_path, output_dir):
    """
    将无表头的CSV文件转换为Parquet格式数据集，并明确指定为validation split

    参数:
        csv_path: 输入的CSV文件路径（无表头，列顺序为：问题,A,B,C,D,答案）
        output_dir: 输出目录（将保存为Hugging Face数据集格式）
    """
    # 1. 读取CSV文件（无表头）
    print(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    # 2. 添加规范的列名
    df.columns = ["question", "A", "B", "C", "D", "answer"]
    print(f"找到 {len(df)} 条数据")

    # 3. 转换为Hugging Face Dataset格式
    dataset = Dataset.from_pandas(df)

    # 4. 创建DatasetDict并指定为validation split
    dataset_dict = DatasetDict({"validation": dataset})

    # 5. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 6. 保存完整数据集（Hugging Face格式）
    print(f"正在保存数据集到: {output_dir}")
    dataset_dict.save_to_disk(output_dir)

    # 7. 单独保存validation split为Parquet文件（可选）
    validation_parquet_path = os.path.join(output_dir, "validation.parquet")
    dataset_dict["validation"].to_parquet(validation_parquet_path)
    print(f"单独保存的Parquet文件: {validation_parquet_path}")
    return dataset_dict

# 使用示例
if __name__ == "__main__":
    # 输入输出配置
    input_csv = "output_filtered.csv"  # 替换为你的CSV文件路径
    output_dir = "YOUR_OUTPUT_PATH"  # 输出目录

    # 执行转换
    dataset = convert_csv_to_parquet_dataset(input_csv, output_dir)

    # 打印验证信息
    print("\n转换结果验证:")
    print(f"数据集结构: {dataset}")
    print(f"Validation split样本数: {len(dataset['validation'])}")
    print(f"首条数据示例: {dataset['validation'][0]}")
```

这样可以把`csv`文件转换为`Dataset`数据集形式。

### 3.2 创建数据集配置文件

在 `/lm-evaluation-harness/lm_eval/tasks` 下创建一个文件夹，命名为 `YOUR_DATASET_NAME`，在这个文件夹下创建一个 `YOUR_DATASET_NAME.yaml`，内容为：

```yaml
task: YOUR_DATASET_NAME
dataset_path: YOUR_DATASET_PATH_FOLDER
test_split: validation
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案："
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: "{{['A', 'B', 'C', 'D'].index(answer)}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 0.0
```

更多创建方式可以参考tasks文件夹下的cmmlu的[yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/cmmlu/_default_template_yaml)构建。

### 3.3 测试精度

```bash
lm_eval --model local-completions   --tasks YOUR_DATASET_NAME   --output_path path/to/save/output  --log_samples   --model_args
  '{
    "model": "your model name",
    "base_url": "http://127.0.0.1:port/v1/completions",
    "tokenizer": "model path",
    "config": "model path",
    "use_fast_tokenizer": true,
    "num_concurrent": 1,
    "max_retries": 3,
    "tokenized_requests": false
  }'
```

## 4. 使用opencompass进行评测

### 4.1 准备数据集

下载数据集并解压至opencompass根目录

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

### 4.2 设置config文件和运行脚本

模型的config文件设置参考[text](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/accelerator_intro.html#api)

修改`path`为部署模型的名字，修改`openai_api_base`为部署模型的`url`，配置模型的`tokenizer_path`，`batch_size`可以适当调大来加速。

数据集一般不用自己配置，参考[text](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/accelerator_intro.html#api)获得推荐配置，或是在每个数据集的config路径下查找合适的配置。例如bbh（big bench hard）数据集，在`opencompass/opencompass/configs/datasets/bbh/`下有`bbh_gen_ee62e9.py`、`bbh_0shot_nocot_academic_gen.py`等， 分别是zero-shot和five-shot的配置，根据需要自由选择。

运行脚本参考[eval_api_demo.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_api_demo.py)进行修改, 导入需要评测的模型配置和需要测试的数据集即可

**可能遇到的报错**：

```bash
Traceback (most recent call last):
...
/mmengine/config/lazy.py", line 205, in __call__
    raise RuntimeError()
RuntimeError
```

解决方法：把该文件```with open(os.path.join(hard_coded_path, 'lib_prompt', f'{_name}.txt'), 'r') as f:```中的地址硬编码为如下：

```bash
hard_coded_path = '/path/to/datasets/bbh' \
        + '/lib_prompt/' \
        + f'{_name}.txt'
```

### 4.3 启动评测

启动评测时运行以下命令：

```bash
opencompass /path/to/your/scripts
```

如果需要额外的参数设置，参考[text](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html)进行额外配置即可。
