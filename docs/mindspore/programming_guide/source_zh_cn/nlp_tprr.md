# 多跳知识推理问答模型TPRR

`Ascend` `自然语言处理` `全流程`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/nlp_tprr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

TPRR(Thinking Path Re-Ranker)
是由华为提出的基于开放域多跳问答的通用模型，用以实现多跳知识推理问答。传统问答中，模型只需要找到与原文中问题相关的句子就可以找到答案。多跳知识推理问答中的问题，需要多次“跳转”才能找到答案。具体来说，给定一个问题，模型需要通过知识从多个相关的文档中推理得到正确回答。TPRR模型分为三个模块：Retriever(检索器)、Reranker(重排器)、Reader(阅读器)。其中Retriever根据给定多跳问题，在百万wiki文档中筛选出包含答案的候选文档序列，Reranker从候选文档序列中筛选出最佳文档序列，最后Reader从最佳文档的多个句子中解析出答案，完成多跳知识推理问答。TPRR模型利用条件概率对完整的推理路径进行建模，并且在训练中引入“思考”的负样本选择策略，在国际权威的HotpotQA评测Fullwiki Setting中荣登榜首，并且在联合准确率、线索准确率等四项指标均达到第一。相比于传统的多跳问答模型，TPRR仅利用纯文本信息而不需要额外的实体抽取等技术，使用MindSpore混合精度特性对TPRR模型进行框架加速，结合Ascend，能获得显著的性能提升。

本篇教程将主要介绍如何在Ascend上，使用MindSpore构建并运行多跳知识推理问答模型TPRR。
> 你可以在这里下载完整的示例代码：
<https://gitee.com/mindspore/models/tree/master/research/nlp/tprr> 。

示例代码目录结构如下：

```text
.
└─tprr
  ├─README.md
  ├─scripts
  | ├─run_eval_ascend.sh                      # Launch retriever evaluation in ascend
  | └─run_eval_ascend_reranker_reader.sh      # Launch re-ranker and reader evaluation in ascend
  |
  ├─src
  | ├─build_reranker_data.py                  # build data for re-ranker from result of retriever
  | ├─config.py                               # Evaluation configurations for retriever
  | ├─hotpot_evaluate_v1.py                   # Hotpotqa evaluation script
  | ├─onehop.py                               # Onehop model of retriever
  | ├─onehop_bert.py                          # Onehop bert model of retriever
  | ├─process_data.py                         # Data preprocessing for retriever
  | ├─reader.py                               # Reader model
  | ├─reader_albert_xxlarge.py                # Albert-xxlarge module of reader model
  | ├─reader_downstream.py                    # Downstream module of reader model
  | ├─reader_eval.py                          # Reader evaluation script
  | ├─rerank_albert_xxlarge.py                # Albert-xxlarge module of re-ranker model
  | ├─rerank_and_reader_data_generator.py     # Data generator for re-ranker and reader
  | ├─rerank_and_reader_utils.py              # Utils for re-ranker and reader
  | ├─rerank_downstream.py                    # Downstream module of re-ranker model
  | ├─reranker.py                             # Re-ranker model
  | ├─reranker_eval.py                        # Re-ranker evaluation script
  | ├─twohop.py                               # Twohop model of retriever
  | ├─twohop_bert.py                          # Twohop bert model of retriever
  | └─utils.py                                # Utils for retriever
  |
  ├─retriever_eval.py                         # Evaluation net for retriever
  └─reranker_and_reader_eval.py               # Evaluation net for re-ranker and reader
```

整体执行流程如下：

1. 准备HotpotQA Development数据集，加载处理数据；
2. 下载训练好的模型文件；
3. 设置TPRR模型参数；
4. 初始化TPRR模型；
5. 加载数据集和模型CheckPoint并进行推理，查看结果及保存输出。

## 准备环节

### 安装软件依赖

1. 安装MindSpore

   实践前，确保已经正确安装MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)安装。

2. 安装transformers（建议版本3.4.0）

    ```bash
    pip install transformers==3.4.0
    ```

### 准备数据

本教程使用的数据是预处理过的[en-Wikipedia](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths/tree/master/retriever)和[HotpotQA
Development数据集](https://hotpotqa.github.io/)。请先下载[预处理数据](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/tprr/data.zip)，解压后放到scripts目录下。

### 准备模型文件

下载模型文件(https://download.mindspore.cn/model_zoo/research/nlp/tprr/)，在scripts文件夹下创建ckpt文件夹，并将下载的模型文件放在ckpt文件夹下，文件目录结构如下：

```text
.
└─tprr
  ├─README.md
  |
  ├─scripts
  | ├─data
  | ├─ckpt
  | | ├─onehop_new.ckpt
  | | ├─onehop_mlp.ckpt
  | | ├─twohop_new.ckpt
  | | ├─twohop_mlp.ckpt
  | | ├─rerank_alberet.ckpt
  | | ├─rerank_downstream.ckpt
  | | ├─reader_alberet.ckpt
  | | ├─reader_downstream.ckpt
  | | |
  | | ├─albert-xxlarge
  | | | ├─config.json
  | | | └─spiece.model
  | | └─
  | ├─run_eval_ascend.sh                      # Launch retriever evaluation in ascend
  | └─run_eval_ascend_reranker_reader.sh      # Launch re-ranker and reader evaluation in ascend
  |
  └─src
```

## 加载数据

Retriever模块加载wiki和HotpotQA预处理的数据文件，通过给定的多跳问题从文档数据中检索出相关文档，加载数据部分在源码的`src/process_data.py`脚本中。

```python
def load_data(self):
    """load data"""
    print('**********************  loading data  ********************** ')
    # wiki data
    f_wiki = open(self.wiki_path, 'rb')
    # hotpotqa dev data
    f_train = open(self.dev_path, 'rb')
    # doc data
    f_doc = open(self.dev_data_path, 'rb')
    data_db = pkl.load(f_wiki, encoding="gbk")
    dev_data = json.load(f_train)
    q_doc_text = pkl.load(f_doc, encoding='gbk')
    return data_db, dev_data, q_doc_text
```

Retriever检索得到的结果保存在scripts目录下，Reranker模块根据该结果，使用自定义的DataGenerator类加载wiki和HotpotQA预处理的数据文件，得到重排序结果，并将其保存在scripts目录下。Reader模块根据重排序结果，同样使用自定义的DataGenerator类加载wiki和HotpotQA预处理的数据文件，提取答案和证据。自定义的DataGenerator类在源码的`src/rerank_and_reader_data_generator.py`脚本中。

```python
class DataGenerator:
    """data generator for reranker and reader"""

    def __init__(self, feature_file_path, example_file_path, batch_size, seq_len,
                 para_limit=None, sent_limit=None, task_type=None):
        """init function"""
        self.example_ptr = 0
        self.bsz = batch_size
        self.seq_length = seq_len
        self.para_limit = para_limit
        self.sent_limit = sent_limit
        self.task_type = task_type
        self.feature_file_path = feature_file_path
        self.example_file_path = example_file_path
        self.features = self.load_features()
        self.examples = self.load_examples()
        self.feature_dict = self.get_feature_dict()
        self.example_dict = self.get_example_dict()
        self.features = self.padding_feature(self.features, self.bsz)
```

## 定义网络

### 设置模型参数

模型参数中用户可以自定义设置topk及onehop_num等参数。topk表示Retriever排序后候选一跳文档个数，topk越大，候选文档越多，召回率提高但会引入更多噪声，准确率下降；onehop_num表示一跳候选文档作为二跳待选文档的数目，onehop_num越大，二跳待选文档越多，召回率提高但会引入更多噪声，准确率下降。

```python
def ThinkRetrieverConfig():
    """retriever config"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_len", type=int, default=64, help="max query len")
    parser.add_argument("--d_len", type=int, default=192, help="max doc len")
    parser.add_argument("--s_len", type=int, default=448, help="max seq len")
    parser.add_argument("--in_len", type=int, default=768, help="in len")
    parser.add_argument("--out_len", type=int, default=1, help="out len")
    parser.add_argument("--num_docs", type=int, default=500, help="docs num")
    parser.add_argument("--topk", type=int, default=8, help="top num")
    parser.add_argument("--onehop_num", type=int, default=8, help="onehop num")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device_num", type=int, default=8, help="device num")
    parser.add_argument("--vocab_path", type=str, default='../vocab.txt', help="vocab path")
    parser.add_argument("--wiki_path", type=str, default='../db_docs_bidirection_new.pkl', help="wiki path")
    parser.add_argument("--dev_path", type=str, default='../hotpot_dev_fullwiki_v1_for_retriever.json',
                        help="dev path")
    parser.add_argument("--dev_data_path", type=str, default='../dev_tf_idf_data_raw.pkl', help="dev data path")
    parser.add_argument("--onehop_bert_path", type=str, default='../onehop.ckpt', help="onehop bert ckpt path")
    parser.add_argument("--onehop_mlp_path", type=str, default='../onehop_mlp.ckpt', help="onehop mlp ckpt path")
    parser.add_argument("--twohop_bert_path", type=str, default='../twohop.ckpt', help="twohop bert ckpt path")
    parser.add_argument("--twohop_mlp_path", type=str, default='../twohop_mlp.ckpt', help="twohop mlp ckpt path")
    parser.add_argument("--q_path", type=str, default='../queries', help="queries data path")
    return parser.parse_args()
```

### 定义模型

定义Retriever模块并加载模型参数。

```python
def evaluation():
    model_onehop_bert = ModelOneHop()
    param_dict = load_checkpoint(config.onehop_bert_path)
    load_param_into_net(model_onehop_bert, param_dict)
    model_twohop_bert = ModelTwoHop()
    param_dict2 = load_checkpoint(config.twohop_bert_path)
    load_param_into_net(model_twohop_bert, param_dict2)
    onehop = OneHopBert(config, model_onehop_bert)
    twohop = TwoHopBert(config, model_twohop_bert)
```

定义Reranker模块并加载模型参数。

```python
    reranker = Reranker(batch_size=batch_size,
                        encoder_ck_file=encoder_ck_file,
                        downstream_ck_file=downstream_ck_file)
```

定义Reader模块并加载模型参数。

```python
    reader = Reader(batch_size=batch_size,
                    encoder_ck_file=encoder_ck_file,
                    downstream_ck_file=downstream_ck_file)
```

## 推理网络

### 运行脚本

调用scripts目录下的shell脚本，启动推理进程。 使用以下命令运行Retriever模块推理脚本，得到的检索结果文件doc_path保存在scripts目录下：

```bash
sh run_eval_ascend.sh
```

Retriever模块推理脚本运行完成后，使用以下命令运行Reranker和Reader模块推理脚本：

```bash
sh run_eval_ascend_reranker_reader.sh
```

推理完成后，结果保存到scripts/eval/目录下的log文件中，可以在对应log文件中查看测评结果。

Retriever模块测评结果：其中val表示找对答案文档的问题数目，count表示问题总数目，PEM表示问题相关文档排序后top-8文档的精确匹配的准确率。

```text
# match query num
val:6959
# query num
count:7404
# one hop match query num
true count: 7112
# top8 paragraph exact match
PEM: 0.9398973527822798
# top8 paragraph exact match in recall
true top8 PEM: 0.9784870641169854
# evaluation time
evaluation time (h): 1.819070938428243
```

Reranker和Reader模块测评结果，其中total_top1_pem表示重排序之后top-1路径的精确匹配的准确率，joint_em表示预测的答案和证据的精确匹配的联合准确率，joint_f1表示预测的答案和证据的联合f1
score。

```text
# top8 paragraph exact match
total top1 pem: 0.8803511141120864
...

# answer exact match
em: 0.67440918298447
# answer f1
f1: 0.8025625656569652
# answer precision
prec: 0.8292800393689271
# answer recall
recall: 0.8136908451841731
# supporting facts exact match
sp_em: 0.6009453072248481
# supporting facts f1
sp_f1: 0.844555664157302
# supporting facts precision
sp_prec: 0.8640844345841021
# supporting facts recall
sp_recall: 0.8446123918845106
# joint exact match
joint_em: 0.4537474679270763
# joint f1
joint_f1: 0.715119580346802
# joint precision
joint_prec: 0.7540052057184267
# joint recall
joint_recall: 0.7250240424067661
```

## 引用

1. Yang Z , Qi P , Zhang S , et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering[C]//Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018.
2. Asai A , Hashimoto K , Hajishirzi H , et al. Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering[J]. 2019.
