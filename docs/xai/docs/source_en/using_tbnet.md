# Using TB-Net Whitebox Recommendation Model

<a href="https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_en/using_tbnet.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## What is TB-Net

TB-Net is a white box recommendation model, which constructs subgraphs in knowledge graphs based on the interaction between users and items as well as the features of items, and then calculates paths in the graphs using a bidirectional conduction algorithm. Finally, we can obtain explainable recommendation results.

Paper: Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

## Preparations

### Downloading Data Package

First of all, we have to download the data package and put it underneath the `models/whitebox/tbnet` directory of a local XAI [source package](https://gitee.com/mindspore/xai):

```bash
wget https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/xai/tbnet_data.tar.gz
tar -xf tbnet_data.tar.gz

git clone https://gitee.com/mindspore/xai.git
mv data xai/models/whitebox/tbnet
```

`xai/models/whitebox/tbnet/` files:

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

- `steam/`: Steam user purchase history dataset.
- `LICENSE`: License text.
- `config.json`: Hyper-parameters and training configuration.
- `src_infer.csv`: Source datafile for inference.
- `src_test.csv`: Source datafile for evaluation.
- `src_train.csv`: Source datafile for training.
- `dataset.py`: Dataset loader.
- `embedding.py`: Embedding module.
- `metrics.py`: Evaluation metrics.
- `path_gen.py`: Data preprocessor.
- `recommend.py`: Result aggregator.
- `tbnet.py`: TB-Net architecture.
- `eval.py`: Example of evaluation.
- `export.py`: Example of exporting trained model.
- `infer.py`: Example of inference.
- `preprocess.py`: Example of data pre-processing.
- `train.py`: Example of training.
- `tbnet_config`: Configuration reader for the examples.

### Preparing Python Environment

TB-Net is part of the XAI package, no extra installation is required besides [MindSpore](https://mindspore.cn/install/en) and [XAI](https://www.mindspore.cn/xai/docs/en/master/installation.html). GPUs are supported.

## Data Pre-processing

The complete example code of this step is [preprocess.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/preprocess.py).

Before training the TB-Net, we have to convert the source datafile to relation path data.

### Source Datafile Format

The source datafiles of the steam dataset all share the exact same CSV format with headers:

`user,item,rating,developer,genre,category,release_year`

The first 3 columns must be present with specific order and meaning:

- `user`: String, user ID, records of the same user must be grouped in consecutive rows in a single file. Splitting the records across different files will give misleading results.
- `item`: String, item ID.
- `rating`: Character, either `c`(user had interactions (e.g. clicked) with the item but not purchased), `p`(user purchased the item) or `x`(other items).

(Remark: There is no `c` rating item in the steam dataset.)

Since the order and meaning of these columns are fixed, the names do not matter, users may choose other names like `uid,iid,act`, etc.

The later columns `developer,genre,category,release_year` are for the item's string attribute IDs. Users should decide the column names (i.e. relation names) and keep them consistent in all source datafiles. There must be at least one attribute column with no maximum limit. In some cases, there are more than one values in each attribute, they should be separated by `;`. Leaving the attribute blank means the item has no such attribute.

The content of source datafiles for different purposes are slightly different:

- `src_train.csv`: For training, the numbers of rows of `p` rating and `c` + `x` rating items should be roughly the same by re-sampling, there is no need to list all items in every user.
- `src_test.csv`: For evaluation, very similar to `src_train.csv` but with less amount of data.
- `src_infer.csv`: For inference, must contain data of ONLY ONE user. ALL `c`, `p` and `x` rating items should be listed. In [preprocess.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/preprocess.py), only the `c` and `x` items are put as recommendation candidates in path data.

### Converting to Relation Path Data

```python
import io
import json
from src.path_gen import PathGen

path_gen = PathGen(per_item_paths=39)

path_gen.generate("./data/steam/src_train.csv", "./data/steam/train.csv")

# save id maps for the later use by Recommender for inference
with io.open("./data/steam/id_maps.json", mode="w", encoding="utf-8") as f:
    json.dump(path_gen.id_maps(), f, indent=4)

# treat newly met items and references in src_test.csv and src_infer.csv as unseen entities
# dummy internal id 0 will be assigned to them
path_gen.grow_id_maps = False

path_gen.generate("./data/steam/src_test.csv", "./data/steam/test.csv")

# for inference, only take interacted('c') and other('x') items as candidate items,
# the purchased('p') items won't be recommended.
# assume there is only one user in src_infer.csv
path_gen.subject_ratings = "cx"

path_gen.generate("./data/steam/src_infer.csv", "./data/steam/infer.csv")
```

`PathGen` is responsible for converting source datafile into relation path data.

### Relation Path Data Format

Relation path data are header-less CSV (all integer values), with columns:

`<subject item>,<label>,<relation1>,<reference>,<relation2>,<historical item>,...`

- `subject item`: Internal id of the sample item for training or the recommendation candidate item for inference.`0` represents unseen items.
- `label`: Ground truth label, `0` - not purchased, `1` - purchased. Ignored when inferencing.
- `relation1`: Internal id of the relation 1 (column name of the subject item's attribute).
- `reference`: Internal id of the reference entity (the common attribute ID).`0` represents unseen references.
- `relation2`: Internal id of the relation 2 (column name of the historical item's attribute).
- `historical item`: Internal id of the historical (`c` or `p` rating) item.`0` represents unseen items.

The column sequence `<relation1>,<reference>,<relation2>,<historical item>` is going to repeat `per_item_paths` times in each row.

### Running preprocess.py

Have to change the current directory to `xai/models/whitebox/tbnet` first.

```bash
python preprocess.py
```

Dataset `./data/stream` will be processed with path data `train.csv`, `test.csv` and `infer.csv` and the source-internal ID maps `id_maps.json` generated in the dataset directory.

## Training and Evaluation

The complete example code of this step is [train.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/train.py).

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

We can see from the above code snap of [train.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/train.py) that `tbnet.py`, `metrics.py` and `dataset.py` provides all the classes for training TB-Net like other MindSpore models. It trains with dataset created from `./data/steam/train.csv` and `./data/steam/test.csv` (for evaluation).

### Running train.py

Have to change the current directory to `xai/models/whitebox/tbnet` first.

```bash
python train.py
```

20 epochs will be trained with the steam dataset and checkpoints will be saved to `./checkpoints/steam` for each epoch.

## Inference

The complete example code of this step is [infer.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/infer.py).

```python
from src.tbnet import TBNet
from src.recommend import Recommender
from src.dataset import create_dataset

...

print(f"creating dataset from {data_path}...")
infer_ds = create_dataset(data_path, cfg.per_item_paths)
infer_ds = infer_ds.batch(cfg.batch_size)

print("inferring...")
# infer and aggregate results

with io.open(id_maps_path, mode="r", encoding="utf-8") as f:
    id_maps = json.load(f)
recommender = Recommender(network, id_maps, args.items)

for item, rl1, ref, rl2, hist_item, _ in infer_ds:
    # assume there is data of only one user in infer_ds and all items inside are candidates.
    recommender(item, rl1, ref, rl2, hist_item)

# show recommendations with explanations
suggestions = recommender.suggest()
for suggest in suggestions:
    print("")
    print(f'Recommends item:"{suggest.item}" (score:{suggest.score}) because:')
    # show explanations
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

We can see from the above code snap of [infer.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/infer.py) that class `Recommender` aggregates all inference results from TB-Net and gives top-k item recommendations. Each recommended item comes with a sorted list of relation paths beginning from the most important one. All IDs and relation names returned by `Recommender` are from `./data/steam/id_maps.json` that originated from the source datafile `./data/steam/src_train.csv`.

### Running infer.py

Have to change the current directory to `xai/models/whitebox/tbnet` first.

```bash
python infer.py --checkpoint_id 19
```

The last checkpoint `./checkpoints/steam/tbnet_epoch19.ckpt` from [Running train.py](#running-train-py) will be used to infer on `./data/steam/infer.csv`. 3 recommended items with 3 relation path explanations each will be given.

## Exporting Trained Model

The complete example code of this step is [export.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/export.py).

```python
from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export
from src.tbnet import TBNet

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

The above code snap of [export.py](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/export.py) shows it is straightforward with [mindspore.export](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.export.html?highlight=export#mindspore.export).

### Running export.py

Have to change the current directory to `xai/models/whitebox/tbnet` first.

```bash
python export.py --config_path ./data/steam/config.json --checkpoint_path ./checkpoints/steam/tbnet_epoch19.ckpt
```

The trained model will be exported as `./tbnet.mindir`.

## Example Script Arguments and Model Performance

For the detail descriptions of the example script arguments and model performance, please refer to [README.md](https://gitee.com/mindspore/xai/blob/master/models/whitebox/tbnet/README.md).
