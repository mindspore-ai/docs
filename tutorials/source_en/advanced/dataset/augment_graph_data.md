# Graph Data Loading and Processing

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/source_en/advanced/dataset/augment_graph_data.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

The `mindspore.dataset` module provided by MindSpore helps users build dataset objects and read text data in batches. In addition, data processing and tokenization operators are built in each dataset class. In this way, data can continuously flow to the training system during training, improving the data training effect.

The following briefly demonstrates how to use MindSpore to load and process graph data.

## Concept of Graph

The basic concept of graphs is introduced to help users better understand graph data reading and augmentation. Generally, a graph (`G`) consists of a series of vertices (`V`) and edges (`E`). Each edge is connected to two nodes in the graph. The formula is as follows:

$$G = F(V, E)$$

The following figure shows a simple graph.

![basicGraph.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/tutorials/source_zh_cn/advanced/dataset/images/basic_graph.png)

The graph includes nodes V = {a, b, c, d} and edges E = {(a, b), (b, c), (c, d), (d, b)}. A connection relationship in the graph usually needs to be described in a mathematical manner. For example, in an adjacency matrix, a matrix C used to describe the connection relationship in the graph is as follows, where a, b, c and d correspond to nodes 1, 2, 3, and 4.

$$
C=
\begin{bmatrix}
1&1&0&0\\
1&1&1&1\\
0&1&1&1\\
0&1&1&1\\
\end{bmatrix}
$$

## Preparing a Dataset

1. Dataset Description

    Common graph datasets include Cora, Citeseer, and PubMed. The following uses Cora as an example.

    > The original dataset can be downloaded from [UCSC](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz). Here, the [preprocessed version](https://github.com/kimiyoung/planetoid)[1] provided by kimiyoung is used.

    The main part of the Cora dataset (`cora.content`) contains 2708 samples, each of which describes information about one scientific paper. These papers are classified into seven categories. Each piece of sample data consists of three parts: a paper number, a word vector of the paper (a 1433-bit binary), and a category of the paper. The reference dataset part (`cora.cites`) contains 5429 lines, and each line contains two paper numbers, indicating that the second paper references the first paper.

2. Download a dataset.

    The following sample code is used to download and decompress the Cora dataset to a specified location:

    ```python
    import os
    import shutil

    if not os.path.exists("./cora"):
        os.mkdir("./cora")
        if not os.path.exists("./planetoid"):
            os.system("git clone https://github.com/kimiyoung/planetoid")
        content = os.listdir("./planetoid/data")
        new_content = []
        for name in content:
            if "cora" in name:
                new_content.append(name)
        for name in new_content:
            path = "./planetoid/data/"+name
            shutil.copy(path, "./cora")
    ```

    The following shows the directory for storing the preprocessed Cora dataset.

    ```text
    ./cora
    ├── ind.cora.allx
    ├── ind.cora.ally
    ├── ind.cora.graph
    ├── ind.cora.test.index
    ├── ind.cora.tx
    ├── ind.cora.ty
    ├── ind.cora.x
    ├── ind.cora.y
    ├── trans.cora.graph
    ├── trans.cora.tx
    ├── trans.cora.ty
    ├── trans.cora.x
    └── trans.cora.y
    ```

3. Convert the dataset format.

    Use the conversion script provided by the \`models\` repository to convert the dataset to the MindSpore Record format. The generated MindSpore Record file is stored in `./cora_mindrecord`.

    ```python
    if not os.path.exists("./cora_mindrecord"):
        os.mkdir("./cora_mindrecord")
        os.system('git clone https://gitee.com/mindspore/models.git')
        os.system('python models/utils/graph_to_mindrecord/writer.py --mindrecord_script cora --mindrecord_file "./cora_mindrecord/cora_mr" --mindrecord_partitions 1 --mindrecord_header_size_by_bit 18 --mindrecord_page_size_by_bit 20 --graph_api_args "./cora"')
    ```

## Loading the Dataset

Currently, MindSpore supports loading of classic datasets used in the text field and datasets in multiple data storage formats. You can also build customized dataset classes to implement customized data loading.

The following demonstrates how to use the `MindDataset` class in the `MindSpore.dataset` module to load the Cora dataset in the MindSpore Record format.

1. Configure a dataset directory and create a dataset object.

    ```python
    import mindspore.dataset as ds
    import numpy as np

    data_file = "./cora_mindrecord/cora_mr"
    dataset = ds.GraphData(data_file)
    ```

2. Access the corresponding API to obtain the graph information, features, and label content.

    ```python
    # View the structure information in the graph.
    graph = dataset.graph_info()
    print("graph info:", graph)

    # Obtain information about all nodes.
    nodes = dataset.get_all_nodes(0)
    nodes_list = nodes.tolist()
    print("node shape:", len(nodes_list))

    # Obtain the feature and label information. A total of 2708 records are displayed.
    # The feature information in each piece of data is a binary representation of 1433 characters used to describe the paper i. The label information refers to the category of the paper.
    raw_tensor = dataset.get_node_feature(nodes_list, [1, 2])
    features, labels = raw_tensor[0], raw_tensor[1]

    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    print("labels:", labels)
    ```

    ```text
        graph info: {'node_type': [0], 'edge_type': [0], 'node_num': {0: 2708}, 'edge_num': {0: 10858}, 'node_feature_type': [1, 2], 'edge_feature_type': []}
        node shape: 2708
        features shape: (2708, 1433)
        labels shape: (2708,)
        labels: [3 4 4 ... 3 3 3]
    ```

## Data Processing

The following demonstrates how to build a pipeline and perform operations such as sampling on nodes.

1. Obtain neighboring nodes of a node to build an adjacency matrix.

    ```python
    neighbor = dataset.get_all_neighbors(nodes_list, 0)

    # The first column of neighbor is node_id, and the second to last columns store the neighboring nodes in the first column. If there are not so many neighboring nodes, fill them with -1.
    print("neighbor:\n", neighbor)
    ```

    ```text
        neighbor:
        [[   0  633 1862 ...   -1   -1   -1]
        [   1    2  652 ...   -1   -1   -1]
        [   2 1986  332 ...   -1   -1   -1]
        ...
        [2705  287   -1 ...   -1   -1   -1]
        [2706  165 2707 ...   -1   -1   -1]
        [2707  598  165 ...   -1   -1   -1]]
    ```

2. Build the adjacency matrix according to the neighboring node information.

    ```python
    nodes_num = labels.shape[0]
    node_map = {node_id: index for index, node_id in enumerate(nodes_list)}
    adj = np.zeros([nodes_num, nodes_num], dtype=np.float32)

    for index, value in np.ndenumerate(neighbor):
        # The first column of neighbor is node_id, and the second to last columns store the neighboring nodes in the first column. If there are not so many neighboring nodes, fill them with -1.
        if value >= 0 and index[1] > 0:
            adj[node_map[neighbor[index[0], 0]], node_map[value]] = 1

    print("adj:\n", adj)
    ```

    ```text
        adj:
        [[0. 0. 0. ... 0. 0. 0.]
        [0. 0. 1. ... 0. 0. 0.]
        [0. 1. 0. ... 0. 0. 0.]
        ...
        [0. 0. 0. ... 0. 0. 0.]
        [0. 0. 0. ... 0. 0. 1.]
        [0. 0. 0. ... 0. 1. 0.]]
    ```

3. Perform node sampling. Common methods such as multi-hop sampling and random walk sampling are supported.

    Figure (a) shows the multi-hop neighborhood-based node neighbor sampling. A sampled node is used as the start point of the next sampling. Figure (b) shows the random walk-based node neighbor sampling. A path is randomly selected to traverse neighboring nodes in sequence. In the corresponding figure, a walk path from V<sub>i</sub> to V<sub>j</sub> is selected.

    ![graph](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/tutorials/source_zh_cn/advanced/dataset/images/graph_sample.png)

    ```python
    # Multi-hop neighborhood-based node neighbor sampling
    neighbor = dataset.get_sampled_neighbors(node_list=nodes_list[0:21], neighbor_nums=[2], neighbor_types=[0])
    print("neighbor:\n", neighbor)

    # Random walk-based node neighbor sampling
    meta_path = [0]
    walks = dataset.random_walk(nodes_list[0:21], meta_path)
    print("walks:\n", walks)
    ```

    ```text
        neighbor:
        [[   0 1862  633]
        [   1  654    2]
        [   2 1666    1]
        [   3 2544 2544]
        [   4 1256 1761]
        [   5 1659 1629]
        [   6 1416  373]
        [   7  208  208]
        [   8  281 1996]
        [   9  723 2614]
        [  10 2545  476]
        [  11 1655 1839]
        [  12 2662 1001]
        [  13 1810 1701]
        [  14 2668 2077]
        [  15 1093 1271]
        [  16 2444  970]
        [  17 2140 1315]
        [  18 2082 1560]
        [  19 1939 1939]
        [  20 2375 2269]]
        walks:
        [[   0 1862]
        [   1  654]
        [   2 1666]
        [   3 2544]
        [   4 2176]
        [   5 1659]
        [   6 1042]
        [   7  208]
        [   8  281]
        [   9  723]
        [  10 2545]
        [  11 1839]
        [  12 2662]
        [  13 1701]
        [  14 2034]
        [  15 1271]
        [  16 2642]
        [  17 2140]
        [  18 2145]
        [  19 1939]
        [  20 2269]]
    ```

    > If the random walk-based node neighbor sampling is used, different results may be displayed during execution.

4. Obtain an edge through a node or obtain a node through an edge.

    ```python
    # Obtain an edge through a node.
    part_edges = dataset.get_all_edges(0)[:10]
    nodes = dataset.get_nodes_from_edges(part_edges)
    print("part edges:", part_edges)
    print("nodes:", nodes)

    # Obtain a node through an edge.
    nodes_pair_list = [(0, 633), (1, 652), (2, 332), (3, 2544)]
    edges = dataset.get_edges_from_nodes(nodes_pair_list)
    print("edges:", edges)
    ```

    ```text
        part edges: [0 1 2 3 4 5 6 7 8 9]
        nodes: [[   0  633]
        [   0 1862]
        [   0 2582]
        [   1    2]
        [   1  652]
        [   1  654]
        [   2 1986]
        [   2  332]
        [   2 1666]
        [   2    1]]
        edges: [ 0  4  7 11]
    ```

## References

\[1] Yang Z, Cohen W, Salakhudinov R. [Revisiting semi-supervised learning with graph embeddings](http://proceedings.mlr.press/v48/yanga16.pdf).
