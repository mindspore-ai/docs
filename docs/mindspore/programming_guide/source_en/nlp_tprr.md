# Multi-hop Knowledge Reasoning Question-answering Model TPRR

`Ascend` `Natural Language Processing` `Whole Process`

Translator: [longvoyage](https://gitee.com/yuanyanglv)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/nlp_tprr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

TPRR(Thinking Path Re-Ranker) is an open-domain knowledge based multi-hop question-answering model proposed by Huawei, which is used to realize multi-hop knowledge reasoning question-answering. In traditional question-answering, as long as the sentences related to the original question is found by the model, the answer can be found. It requires multiple "jumps" to find the answer for multi-hop knowledge reasoning question. Specifically, the model needs to use knowledge from multiple related documents to infer the correct answer for the given question. There are three modules in TPRR model: Retriever, Reranker and Reader. According to the given multi hop question, Retriever selects the candidate document sequence containing the answer from millions of Wiki documents, Reranker selects the best document sequence from the candidate document sequence, and finally Reader parses the answer from multiple sentences of the best document to complete the multi-hop knowledge reasoning question-answering. TPRR model uses conditional probability to model the complete reasoning path, and introduces the negative sample selection strategy of "thinking" in the training. It ranks first in Fullwiki Setting of international authoritative HotpotQA evaluation, and ranks first in the joint accuracy, clue accuracy and other two indicators. Compared with the traditional multi-hop question-answering model, TPRR only uses pure text information and does not need additional entity extraction technology. MindSpore hybrid precision feature is used to speed up TPRR model from framework. Combined with Ascend, it can achieve significant performance improvement.

This tutorial will mainly introduce how to build and run a multi-hop knowledge reasoning question-answering model TPRR with MindSpore on Ascend.

> You can download the complete sample code here:
<https://gitee.com/mindspore/models/tree/master/research/nlp/tprr>.

The sample code directory structure is as follows:

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

The overall execution process is as follows:

1. Prepare HotpotQA Development dataset, load processing data;
2. Prepare checkpoint for model;
3. Set TPRR model parameters;
4. Initialize the TPRR model;
5. Load the dataset and model CheckPoint and perform inference, check the results and save the output.

## Preparation

### Installing Dependent Software

1. Install MindSpore

   Before practicing, make sure that MindSpore has been installed correctly.If not, you can install it through [the MindSpore installation page](https://www.mindspore.cn/install/en).

2. Install transformers(recommended version is 3.4.0)

    ```bash
    pip install transformers==3.4.0
    ```

### Preparing Data

The data used in this tutorial is the preprocessed [en-Wikipedia](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths/tree/master/retriever) and [HotpotQA Development datasets](https://hotpotqa.github.io/). Please download the [preprocessed data](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/tprr/data.zip) first, then decompress it and place it in '/scripts'.

### Preparing checkpoint

Please download [checkpoint](https://download.mindspore.cn/model_zoo/research/nlp/tprr/), then make directory '/ckpt' in 'scripts' and place downloaded checkpoint files in '/ckpt', directory structure is as following:

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

## Loading Data

Store the downloaded data in the scripts directory. The Retriever module loads the data files preprocessed by wiki and HotpotQA, and retrieves relevant documents from the data according to the given multi-hop question. The source code of data loading is in the file `src/process_data.py`.

```python
def load_data(self):
    """load data"""
    print('**********************  loading data  ********************** ')
    f_wiki = open(self.wiki_path, 'rb')
    f_train = open(self.dev_path, 'rb')
    f_doc = open(self.dev_data_path, 'rb')
    data_db = pkl.load(f_wiki, encoding="gbk")
    dev_data = json.load(f_train)
    q_doc_text = pkl.load(f_doc, encoding='gbk')
    f_wiki.close()
    f_train.close()
    f_doc.close()
    return data_db, dev_data, q_doc_text
```

Retrieved results of the Retriever module are saved in the scripts directory. According to the results, the Reranker  module uses a custom DataGenerator class loading the data files preprocessed by wiki and HotpotQA to generator the reordering results and save them in the scripts directory. According to the reordering results, the Reader module also uses a custom DataGenerator class loading data files preprocessed by wiki and HotpotQA to extract answers and evidence. The source code of custom DataGenerator class is in the file `src/rerank_and_reader_data_generator.py`.

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

## Defining the Network

### Setting Model Parameters

The class ThinkRetrieverConfig for customizing parameters of model is in the script `src/config.py`. The user can customize parameters such as topk and onehop_num in the model. Topk represents the number of candidate one-hop documents after Retriever sorting. The larger the topk, the more candidate documents. The recall rate will increase and more noise will be introduced, the accuracy rate will decrease; Onehop_num represents the number of one-hop candidate documents as two-hop candidate documents. The larger onehop_num, the more documents to be selected for the second hop. The recall rate will increase and more noise will be introduced, the accuracy rate will decrease.

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

### Defining the Model

Define the Retriever module and load the model parameters, the following example code is in the script `retriever_eval.py`.

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

Define the Reranker module and load the model parameters, class Reranker is in the script `src/reranker.py`.

```python
    reranker = Reranker(batch_size=batch_size,
                        encoder_ck_file=encoder_ck_file,
                        downstream_ck_file=downstream_ck_file)
```

Define the Reader module and load the model parameters, class Reader is in the script `src/reader.py`.

```python
    reader = Reader(batch_size=batch_size,
                    encoder_ck_file=encoder_ck_file,
                    downstream_ck_file=downstream_ck_file)
```

## Inference Network

### Running Script

Run the shell script in the scripts directory to start the inference process. Run the script with the following command for the Retriever module, retriever result file 'doc_path' will be saved in '/scripts':

```bash
sh run_eval_ascend.sh
```

After retrieving is completed, run the script with the following command for the Reranker module and Reader module:

```bash
sh run_eval_ascend_reranker_reader.sh
```

After the inference is completed, the result is saved to the log file in `scripts/eval/` directory, and the evaluation result can be checked in the corresponding log file.

Evaluation results of the Retriever module: val represents the number of questions found in the correct answer document, count represents the total number of questions, and PEM represents the accuracy of the top-8 documents after the problem-related documents are sorted.

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

The following is Reranker and Reader module evaluation results, total_top1_pem represents the accuracy of the exact matching of the top-1 path after reordering, joint_em represents the joint accuracy of the predicted answer and the exact match of the evidence, joint_f1 represents the combined f1 score of the predicted answer and the evidence.

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

## Reference

1. Yang Z , Qi P , Zhang S , et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering[C]// Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018.
2. Asai A , Hashimoto K , Hajishirzi H , et al. Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering[J]. 2019.
