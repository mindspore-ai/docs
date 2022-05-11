# 使用 TB-Net 白盒推荐模型

<a href="https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_zh_cn/using_tbnet.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 什么是 TB-Net

TB-Net是一个基于知识图谱的可解释推荐系统，它将用户和商品的交互信息以及物品的属性信息在知识图谱中构建子图，并利用双向传导的计算方法对图谱中的路径进行计算，最后得到可解释的推荐结果。

论文：Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

## 准备

### 下载数据集

首先，我们要下一个用例数据包并解压到一个本地 [XAI原码包](https://gitee.com/mindspore/xai) 中的`models/whitebox/tbnet`文件夹：

```bash
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/xai/tbnet_data.tar.gz
tar -xf tbnet_data.tar.gz

git clone https://gitee.com/mindspore/xai.git
mv data xai/models/whitebox/tbnet
```

`xai/models/whitebox/tbnet/` 文件夹结构：

```bash
xai/models/whitebox/tbnet/
├── README.md
├── README_CN.md
├── data/
│    └── steam/
│         ├── LICENSE
│         ├── config.json
│         ├── src_infer.csv
│         ├── src_test.csv
│         └── src_train.csv
├── src/
|    ├─dataset.py
|    ├─embedding.py
|    ├─metrics.py
|    ├─path_gen.py
|    ├─recommend.py
|    └─tbnet.py
├── eval.py
├── export.py
├── infer.py
├── preprocess.py
├── train.py
└── tbnet_config.py
```

- `steam/`：Steam 用户历史行为数据集。
- `LICENSE`：数据集开原执照。
- `config.json`：模型超参及训练设定。
- `src_infer.csv`：推理用原始数据。
- `src_test.csv`：评估用原始数据。
- `src_train.csv`：训练用原始数据。
- `dataset.py`：数据集加载器。
- `embedding.py`：实体嵌入模组。
- `metrics.py`：模型度量。
- `path_gen.py`：数据预处理器。
- `recommend.py`：推理结果集成器。
- `tbnet.py`：TB-Net 网络架构。
- `eval.py`：评估用例。
- `export.py`：导出已训练模型用例。
- `infer.py`：推理用例。
- `preprocess.py`：数据预处理用例。
- `train.py`：训练用例。
- `tbnet_config`：用例的设定阅读器。

### 准备 Python 环境

TB-Net 是 XAI 的一部份，用户在安装好 [MindSpore](https://mindspore.cn/install) 及 [XAI](https://www.mindspore.cn/xai/docs/zh-CN/master/installation.html) 后即可使用，支持 GPU。

## 数据预处理

本步骤的完整用例代码：[preprocess.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/preprocess.py) 。

在训练 TB-Net 前我们必须把原始数据转换为关系路径数据。

### 原始数据格式

Steam 数据集的所有原始数据文件都拥有完全相同的 CSV 格式，文件头是：

`user,item,rating,developer,genre,category,release_year`

头三个列是必需的，而且它们的次序及意义是固定的：

- `user`：字串，用户ID，同一用户的数据必须被结集在同一个文件中相邻的行，把数据分散在不相邻的行或横跨不同的文件会导致错误的结果。
- `item`：字串，商品ID。
- `rating`：单字符，商品评级，可选：`c`（用户跟该商品有过互动如点击，但没有购买过）、`p`（用户购买过该商品）、`x`（其他商品）。

（备注：Steam 数据集并没有 `c` 评级的商品）

由于以上三个列的次序及意义是固定的，所以用户可以自定义它们的名称，例如 `uid,iid,act` 等。

其余的列 `developer,genre,category,release_year` 是商品的属性（即关系）列，储存字串属性值ID。用户须自定义列的名称（即关系名称）并在所有相关的原始数据文件中保持一致。最少要有一个属性列，但并没有最大数量限制。如果在一个属性中商品具有超过一个的属性值ID，它们必须由`;`分隔。如果商品不具有谋些属性，请把该属性留空。

不同使用目的原始数据文件的具体内容都有一些区别：

- `src_train.csv`：训练用，在总体上，`p` 评级的商品行数要和 `c`、`x` 评级的商品行数之和大致持平，可以使用二次采样达致，无须为每个用户列出所有的商品。
- `src_test.csv`：评估用，跟 `src_train.csv` 一样， 但数据量较少。
- `src_infer.csv`：推理用，只能含有一个用户的数据，而且要把所有 `c`、`p` 及 `x` 评级的商品都列出。在 [preprocess.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/preprocess.py) 中，只有 `c` 或 `x` 评级的商品才会成为关系路径数据中的候选推荐商品。

### 转换为关系路径数据

```python
import io
import json
from src.path_gen import PathGen

path_gen = PathGen(per_item_paths=39)

path_gen.generate("./data/steam/src_train.csv", "./data/steam/train.csv")

# 储存ID映射表以留待推理时 Recommender 使用
with io.open("./data/steam/id_maps.json", mode="w", encoding="utf-8") as f:
    json.dump(path_gen.id_maps(), f, indent=4)

# 把在 src_test.csv 及 src_infer.csv 新遇到的商品及属性ID都视为默生实体，内部ID 0 会用来代表它们
path_gen.grow_id_maps = False

path_gen.generate("./data/steam/src_test.csv", "./data/steam/test.csv")

# src_infer.csv 只含有一个用户的数据，只有 c 或 x 评级的商品才会成为 infer.csv 中的候选推荐商品
path_gen.subject_ratings = "cx"

path_gen.generate("./data/steam/src_infer.csv", "./data/steam/infer.csv")
```

`PathGen` 是负责把原始数据转换为关系路径数据的类。

### 关系路径数据格式

关系路径数据是没有文件头的CSV（全部为整数值），对应的数据列如下：

`<subject item>,<label>,<relation1>,<reference>,<relation2>,<historical item>,...`

- `subject item`：训练样本商品或推理候选推荐商品的内部ID。 `0` 代表默生商品。
- `label`：真实标签， `0` - 没有购买过， `1` - 有购买过，在推理时会被忽略。
- `relation1`：关系1（样本商品的属性列名称）的内部ID。
- `reference`：参照实体 (共同的属性值ID) 的内部ID。 `0` 代表默生参照实体。
- `relation2`：关系2（历史商品的属性列名称）的内部ID。
- `historical item`：历史（`c` 或 `p`评级）商品的内部ID。 `0` 代表默生商品。

数据列序列 `<relation1>,<reference>,<relation2>,<historical item>` 会重覆 `per_item_paths` 那么多次。

### 执行 preprocess.py

必须先把当前路径设为 `xai/models/whitebox/tbnet`。

```bash
python preprocess.py
```

`./data/stream` 数据集会被处理生成关系路径数据 `train.csv`、`test.csv`、`infer.csv`以及原始-内部ID映射表`id_maps.json`。

## 训练及评估

本步骤的完整用例代码：[train.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/train.py) 。

```python
from src.tbnet import TBNet, NetWithLossCell, TrainStepWrapCell, EvalNet
from src.dataset import create_dataset
from src.metrics import AUC, ACC

...

train_ds = create_dataset(train_csv_path, cfg.per_item_paths).batch(cfg.batch_size)
test_ds = create_dataset(test_csv_path, cfg.per_item_paths).batch(cfg.batch_size)

print("creating TBNet for training...")
network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
loss_net = NetWithLossCell(network, cfg.kge_weight, cfg.node_weight, cfg.l2_weight)
train_net = TrainStepWrapCell(loss_net, cfg.lr)
train_net.set_train()
eval_net = EvalNet(network)
time_callback = TimeMonitor(data_size=train_ds.get_dataset_size())
loss_callback = MyLossMonitor()
model = Model(network=train_net, eval_network=eval_net, metrics={'auc': AUC(), 'acc': ACC()})
print("training...")
for i in range(args.epochs):
    print(f'===================== Epoch {i} =====================')
    model.train(epoch=1, train_dataset=train_ds, callbacks=[time_callback, loss_callback], dataset_sink_mode=False)
    train_out = model.eval(train_ds, dataset_sink_mode=False)
    test_out = model.eval(test_ds, dataset_sink_mode=False)
    print(f'Train AUC:{train_out["auc"]} ACC:{train_out["acc"]}  Test AUC:{test_out["auc"]} ACC:{test_out["acc"]}')

    ckpt_path = os.path.join(ckpt_dir_path, f'tbnet_epoch{i}.ckpt')
    save_checkpoint(network, ckpt_path)
    print(f'checkpoint saved: {ckpt_path}')
```

从以上的 [train.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/train.py) 代码可见 `tbnet.py`、`dataset.py` 及 `metrics.py` 提供了所有训练TB-Net所需要的类。代码用 `./data/steam/train.csv` 及 `./data/steam/test.csv` 构造了 `Dataset` 对象以进行训练及评估。

### 执行 train.py

必须先把当前路径设为 `xai/models/whitebox/tbnet`。

```bash
python train.py
```

会使用Steam数据集进行20个epoch的训练，并为每个epoch储存一个checkpoint文件到 `./checkpoints/steam`。

## 推理

本步骤的完整用例代码：[infer.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/infer.py) 。

```python
from src.tbnet import TBNet
from src.recommend import Recommender
from src.dataset import create_dataset

...

print(f"creating dataset from {data_path}...")
infer_ds = create_dataset(data_path, cfg.per_item_paths)
infer_ds = infer_ds.batch(cfg.batch_size)

print("inferring...")
# 推理并收集结果

with io.open(id_maps_path, mode="r", encoding="utf-8") as f:
    id_maps = json.load(f)
recommender = Recommender(network, id_maps, args.items)

for item, rl1, ref, rl2, hist_item, _ in infer_ds:
    # infer_ds 只含有一个用户的数据，并且所有的样本商品均为候选推荐商品
    recommender(item, rl1, ref, rl2, hist_item)

# 显示被推荐的商品及推荐理据
suggestions = recommender.suggest()
for suggest in suggestions:
    print("")
    print(f'Recommends item:"{suggest.item}" (score:{suggest.score}) because:')
    # 显示推荐理据
    explanation = 0
    for path in suggest.paths:
        if path.relation1 == path.relation2:
            print(f'- it shares the same {path.relation1}:"{path.reference}" with user\'s '
                  f'historical item:"{path.hist_item}".\n  (importance:{path.importance})')
        else:
            print(f'- it has {path.relation1}:"{path.reference}" while which is {path.relation2} '
                  f'of user\'s historical item:"{path.hist_item}".\n  (importance:{path.importance})')
        explanation += 1
        if explanation >= args.explanations:
            break
```

从以上的 [infer.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/infer.py) 代码可见 `Recommender` 收集 TB-Net 的推埋结果并给出top-k推荐商品，每个推荐商品都会伴随一个按重要性由高到低排列的关系路径序列作为解释。所有由 `Recommender` 返回的ID及关系名称均源自 `./data/steam/src_train.csv`，并且暂存于 `./data/steam/id_maps.json`。

### 执行 infer.py

必须先把当前路径设为 `xai/models/whitebox/tbnet`。

```bash
python infer.py --checkpoint_id 19
```

[执行 train.py](#执行-train.py) 产生出的最后一个checkpoint文件 `./checkpoints/steam/tbnet_epoch19.ckpt` 会被用作推埋。会给出三个推荐商品，每个推荐商品都会伴随三条最重要的关系路径以作解释。

## 导出已训练模型

本步骤的完整用例代码：[export.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/export.py) 。

```python
from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export
from mindspore_xai.whitebox.tbnet import TBNet

...

network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(network, param_dict)

item = Tensor(np.ones((1,)).astype(np.int))
rl1 = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
ref = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
rl2 = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
his = Tensor(np.ones((1, cfg.per_item_paths)).astype(np.int))
inputs = [item, rl1, ref, rl2, his]
file_name = os.path.realpath(args.file_name)
export(network, *inputs, file_name=file_name, file_format=args.file_format)
```

从以上的 [export.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/export.py) 可见，使用 [mindspore.export](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.export.html?highlight=export#mindspore.export) 导出已训练模型是十分简单直接的。

### 执行 export.py

必须先把当前路径设为 `xai/models/whitebox/tbnet`。

```bash
python export.py --config_path ./data/steam/config.json --checkpoint_path ./checkpoints/steam/tbnet_epoch19.ckpt
```

已训练模型会被导出成 `./tbnet.mindir` 文件。

## 用例脚本参数及模型性能指标

请参考 [README_CN.md](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/README_CN.md) 以了解各个用例脚本的详细参数及模型性能指标。
