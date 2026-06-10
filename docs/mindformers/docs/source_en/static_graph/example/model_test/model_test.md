<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Practice Case: Interconnecting MindSpore Transformers with General Evaluation Tools

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/example/model_test/model_test.md)

This article is contributed by Killjoy, chen-xialei, fuyao-15989607593, laozhuang, and oacjiewen.

During the development of LLMs, after training or fine-tuning models using MindSpore Transformers, users often use general evaluation tools to evaluate the model capabilities on custom datasets. This document describes how to interconnect a deployed MindSpore Transformers model with general evaluation tools. It covers the use of vLLM-MindSpore for model deployment and the evaluation of model capabilities based on two general evaluation frameworks: `lm-eval` and `opencompass`. This practice case helps you understand how to use general evaluation tools to evaluate models trained or fine-tuned using MindSpore Transformers.

## 1. Environment Preparations

You need to install the following environment for model deployment and evaluation.

| Dependent Software          | Version|
|--------------------|----------|
| MindSpore Transformers |    1.6.0    |
| vLLM-MindSpore     | 0.5.0       |
| lm-eval            | 0.4.9       |
| opencompass              | 0.5.0      |

### 1.1 MindSpore Transformers

Set up the environment by referring to [MindSpore Transformers Installation Guidelines](https://www.mindspore.cn/mindformers/docs/en/master/installation.html).

### 1.2 vLLM-MindSpore

Run the following commands to pull the vLLM-MindSpore plugin code repository and build an image:

```bash
git clone https://atomgit.com/mindspore/vllm-mindspore.git
bash build_image.sh
```

> If the image building times out, you can add `ENV UV_HTTP_TIMEOUT=3000` to the `build_image.sh` script and replace the image repository with a faster one in the `install_depend_pkgs.sh` script.

Create a Docker based on the server configuration. For details, see [Docker Installation](https://www.mindspore.cn/vllm_mindspore/docs/en/master/getting_started/installation/installation.html).

### 1.3 lm-eval

**Note: It is strongly recommended that you create a separate conda environment with Python 3.10 or later to avoid compatibility issues.**

Note that you need to use the local installation method and do not directly use `pip install lm-eval`.

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

If the error message `Error: Please make sure the libxml2 and libxslt development packages are installed` is displayed, run the following command for installation:

```bash
conda install -c conda-forge libxml2 libxslt
```

To avoid possible version compatibility issues, install the `datasets` and `transformers` libraries of specific versions.

```bash
pip install datasets==2.18.0
pip install transformers==4.35.2
```

### 1.4 opencompass

```bash
pip install -U opencompass
```

During the installation, the following error may be reported:

```bash
AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
```

Solutions:
Use the [source code](https://github.com/open-compass/opencompass) for installation, and delete `pyext` and `rouge` from the `requirements/runtime.txt` file.

## 2. Model Deployment

Set the environment variable.

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers
```

Deploy the model.

```bash
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model MODEL_PATH --port YOUR_PORT --host 0.0.0.0 --served-model-name YOUR_MODEL_NAME
```

## 3. lm-eval Usage for Evaluation

lm-eval is a large-scale comprehensive evaluation framework, which is applicable to many general-domain test sets (such as MMLU and CEval) and supports convenient custom data tests.

### 3.1 Processing the Dataset

> This step is required for customizing a dataset. When testing a general test set, directly use the [official tutorial](https://github.com/EleutherAI/lm-evaluation-harness).

Assume that there is a custom dataset on the local PC. The dataset is a `CSV` file. Each data record is a single-choice question with the following six attributes: a question, four options, and an answer. The file does not have a table header and the file name is `output_filtered.csv`. Run the following codes to convert the `CSV` file into the `Dataset` format that is suitable for model processing:

```python
import pandas as pd
from datasets import Dataset, DatasetDict
import os

def convert_csv_to_parquet_dataset(csv_path, output_dir):
    """
    Convert the CSV file without a header into a Parquet dataset and specify it as the validation split.

    Parameters:
        csv_path: Path of the input CSV file (without a header, with columns in the order of question, options A, B, C, and D, and answer).
        output_dir: Output directory (saved in the Hugging Face dataset format).
    """
    # 1. Read the CSV file (without a header).
    print(f"Reading the CSV file: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    # 2. Add standard column names.
    df.columns = ["question", "A", "B", "C", "D", "answer"]
    print(f"Found {len(df)} records.")

    # 3. Convert to the Hugging Face dataset format.
    dataset = Dataset.from_pandas(df)

    # 4. Create a DatasetDict and specify it as the validation split.
    dataset_dict = DatasetDict({"validation": dataset})

    # 5. Create the output directory.
    os.makedirs(output_dir, exist_ok=True)

    # 6. Save the complete dataset (in the Hugging Face format).
    print(f"Saving the dataset to: {output_dir}")
    dataset_dict.save_to_disk(output_dir)

    # 7. (Optional) Save the validation split as a Parquet file separately.
    validation_parquet_path = os.path.join(output_dir, "validation.parquet")
    dataset_dict["validation"].to_parquet(validation_parquet_path)
    print(f"Parquet file saved separately: {validation_parquet_path}")
    return dataset_dict

# Examples
if __name__ == "__main__":
    # Input and output configuration.
    input_csv = "output_filtered.csv"  # Replace it with the path of your CSV file.
    output_dir = "YOUR_OUTPUT_PATH"  # Output directory.

    # Execute the conversion.
    dataset = convert_csv_to_parquet_dataset(input_csv, output_dir)

    # Print the verification information.
    print("\n Verification of the conversion result:")
    print(f"Dataset structure: {dataset}")
    print(f"Number of validation split samples: {len(dataset['validation'])}")
    print(f"Example of the first data record: {dataset['validation'][0]}")
```

In this way, the `CSV` file can be converted into the `Dataset` format.

### 3.2 Creating a Dataset Configuration File

Create a folder named `YOUR_DATASET_NAME` under `/lm-evaluation-harness/lm_eval/tasks` and create a `YOUR_DATASET_NAME.yaml` file in the folder. The content is as follows:

```yaml
task: YOUR_DATASET_NAME
dataset_path: YOUR_DATASET_PATH_FOLDER
test_split: validation
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n Answer:"
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

For more creation methods, see the [yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/cmmlu/_default_template_yaml) build of CMMLU in the **tasks** folder.

### 3.3 Testing Precision

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

## 4. OpenCompass Usage for Evaluation

### 4.1 Preparing a Dataset

Download the dataset and decompress it to the root directory of OpenCompass.

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

### 4.2 Setting the config File and Running Script

For details about how to set the **config** file of the model, see [text](https://opencompass.readthedocs.io/en/latest/advanced_guides/accelerator_intro.html).

Change `path` to the name of the deployed model, change `openai_api_base` to the `url` of the deployed model, and set `tokenizer_path` of the model. You can increase the value of `batch_size` for acceleration.

Generally, you do not need to configure the dataset. You can obtain the recommended configuration by referring to [text](https://opencompass.readthedocs.io/en/latest/advanced_guides/accelerator_intro.html) or find the appropriate configuration in the **config** path of each dataset. For example, in the big bench hard (BBH) dataset, there are `bbh_gen_ee62e9.py` and `bbh_0shot_nocot_academic_gen.py` in `opencompass/opencompass/configs/datasets/bbh/`, which are the configurations of **zero-shot** and **five-shot**, respectively. Select the configuration as required.

Modify the script by referring to [eval_api_demo.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_api_demo.py). Import the model configurations and datasets to be tested.

**Possible error**:

```bash
Traceback (most recent call last):
...
/mmengine/config/lazy.py", line 205, in __call__
    raise RuntimeError()
RuntimeError
```

Solution: Hard code the address in the file `with open(os.path.join(hard_coded_path, 'lib_prompt', f'{_name}.txt'), 'r') as f:` as follows:

```bash
hard_coded_path = '/path/to/datasets/bbh' \
        + '/lib_prompt/' \
        + f'{_name}.txt'
```

### 4.3 Starting the Evaluation

Run the following command to start the evaluation:

```bash
opencompass /path/to/your/scripts
```

If additional parameter settings are required, refer to [text](https://opencompass.readthedocs.io/en/latest/advanced_guides/accelerator_intro.html) for additional configurations.
