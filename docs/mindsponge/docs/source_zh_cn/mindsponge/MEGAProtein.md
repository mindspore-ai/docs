# 数据驱动模型 MEGA-Protein

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindscience/docs/source_zh_cn/mindsponge/MEGAProtein.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

使用计算机高效计算获取蛋白质空间结构的过程被称为蛋白质结构预测，传统的结构预测工具一直存在精度不足的问题。直至2020年谷歌DeepMind团队提出[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) [1,2]，该模型相较于传统工具预测精度大幅提升，所得结构与真实结构误差接近实验方法。但是仍存在数据前处理耗时过长、缺少MSA时预测精度不准、缺乏通用评估结构质量工具的问题。针对这些问题，高毅勤老师团队与MindSpore科学计算团队合作进行了一系列创新研究，开发出了更准确和更高效的蛋白质结构预测工具**MEGA-Protein**。

MEGA-Protein主要由三部分组成：

- **蛋白质结构预测工具MEGA-Fold**，网络模型部分与AlphaFold2相同，在数据预处理的多序列对比环节采用了[MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf) [3]进行序列检索，相比于原版端到端速度提升2-3倍；同时借助内存复用大幅提升内存利用效率，同硬件条件下支持更长序列的推理（基于32GB内存的Ascend910运行时最长支持3072长度序列推理）。
- **MSA生成工具MEGA-EvoGen**，能显著提升单序列的预测速度，并且能够在MSA较少（few shot）甚至没有MSA（zero-shot，即单序列）的情况下，帮助MEGA-Fold/AlphaFold2等模型维持甚至提高推理精度，突破了在「孤儿序列」、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制。
- **蛋白质结构评分工具MEGA-Assessement**，该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化。

## 可用的模型和数据集

| 所属模块  | 文件名               | 大小              | 描述                                             | Model URL                                                                                         |
| --------- | -------------------- | ----------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| MEGA-Fold | `MEGA_Fold_1.ckpt` | 356MB             | MEGA-Fold在PSP数据集训练的数据库与checkpoint链接 | [下载链接](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt) |
| PSP       | `PSP`              | 1.6TB(解压后25TB) | PSP蛋白质结构数据集，可用于MEGA-Fold训练         | [下载链接](http://ftp.cbi.pku.edu.cn/psp/)                                                           |

## 环境配置

### 硬件环境与框架

本工具基于[MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)生物计算库与[MindSpore](https://www.mindspore.cn/)AI框架开发，MindSpore 1.8及以后的版本均可运行，MindSpore安装和配置可以参考[MindSpore安装页面](https://www.mindspore.cn/install)。本工具可以在Ascend910或32G以上内存的GPU上运行，默认使用全精度推理，基于Ascend运行时需调用混合精度。

蛋白质结构预测工具MEGA-Fold依赖多序列比对(MSA，multiple sequence alignments)与模板检索生成等传统数据库搜索工具提供的共进化与模板信息，配置数据库搜索需**2.5T硬盘**（推荐SSD）和与Kunpeng920性能持平的CPU。

### 配置数据库检索

- 配置MSA检索

    首先安装MSA搜索工具**MMseqs2**，该工具的安装和使用可以参考[MMseqs2 User Guide](https://mmseqs.com/latest/userguide.pdf)，安装完成后运行以下命令配置环境变量：

    ```shell
    export PATH=$(pwd)/mmseqs/bin/:$PATH
    ```

    然后下载MSA所需数据库：

    - [uniref30_2103](http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz)：压缩包68G，解压后375G
    - [colabfold_envdb_202108](http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz)：压缩包110G，解压后949G

    下载完成后需解压并使用MMseqs2处理数据库，数据处理参考[colabfold](http://colabfold.mmseqs.com)，主要命令如下：

    ```bash
    tar xzvf "uniref30_2103.tar.gz"
    mmseqs tsv2exprofiledb "uniref30_2103" "uniref30_2103_db"
    mmseqs createindex "uniref30_2103_db" tmp1 --remove-tmp-files 1

    tar xzvf "colabfold_envdb_202108.tar.gz"
    mmseqs tsv2exprofiledb "colabfold_envdb_202108" "colabfold_envdb_202108_db"
    mmseqs createindex "colabfold_envdb_202108_db" tmp2 --remove-tmp-files 1
    ```

- 配置模板检索

    首先安装模板搜索工具[**HHsearch**](https://github.com/soedinglab/hh-suite)
    与[**kalign**](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)，然后下载模板检索所需数据库：

    - [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：压缩包19G，解压后56G。
    - [mmcif database](https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/)： 零散压缩文件～50G，解压后～200G，需使用脚本下载，下载后需解压所有mmcif文件放在同一个文件夹内。
    - [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K。

    *数据库下载网站均为国外网站，下载速度可能较慢*。

    - 配置数据库检索config

    根据数据库安装情况配置 `config/data.yaml`中数据库搜索的相关配置 `database_search`，相关参数含义如下：

    ```bash
    # 模板检索配置
    hhsearch_binary_path   HHsearch可执行文件路径
    kalign_binary_path     kalign可执行文件路径
    pdb70_database_path    pdb70文件夹路径
    mmcif_dir              mmcif文件夹路径
    obsolete_pdbs_path     PDB IDs的映射文件路径
    max_template_date      模板搜索截止时间，该时间点之后的模板会被过滤掉，默认值"2100-01-01"
    # 多序列比对MSA配置
    mmseqs_binary          MMseqs2可执行文件路径
    uniref30_path          uniref30文件夹路径
    database_envdb_dir     colabfold_envdb_202108文件夹路径
    a3m_result_path        mmseqs2检索结果(msa)的保存路径，默认值"./a3m_result/"
    ```

## 代码示例

在MindSPONGE代码仓中可获取[MEGAProtein相关代码](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein)。

代码目录：

```bash
├── MEGA-Protein
    ├── main.py                         // MEGA-Protein主脚本
    ├── README.md                       // MEGA-Protein相关英文说明
    ├── README_CN.md                    // MEGA-Protein相关中文说明
    ├── config
        ├── data.yaml                   //数据处理参数配置
        ├── model.yaml                  //模型参数配置
    ├── data
        ├── dataset.py                  // 异步数据读取脚本
        ├── hhsearch.py                 // python封装的HHsearch工具
        ├── kalign.py                   // python封装的Kalign工具
        ├── msa_query.py                // python封装的MSA搜索及处理工具
        ├── msa_search.sh               // 调用MMseqs2搜索MSA的shell脚本
        ├── multimer_pipeline.py        // 复合物数据预处理脚本
        ├── parsers.py                  // mmcif文件读取脚本
        ├── preprocess.py               // 数据预处理脚本
        ├── protein_feature.py          // MSA与template特征搜索与整合脚本
        ├── templates.py                // 模板搜索脚本
        ├── utils.py                    // 数据处理所需功能函数
    ├── examples
        ├── pdb                         //样例输入数据（.pkl文件）
        ├── pkl                         //样例输出数据（.pdb文件）
    ├── model
        ├── fold.py                     // MEGA-Fold主模型脚本
    ├── module
        ├── evoformer.py                // evoformer特征提取模块
        ├── fold_wrapcell.py            // 训练迭代封装模块
        ├── head.py                     // MEGA-Fold附加输出模块
        ├── loss_module.py              // MEGA-Fold训练loss模块
        ├── structure.py                // 3D结构生成模块
        ├── template_embedding.py       // 模板信息提取模块
    ├── scripts
        ├── run_fold_infer_gpu.sh       // GPU运行MEGA-Fold推理示例
        ├── run_fold_train_ascend.sh    // Ascend运行MEGA-Fold推理示例
```

### MEGA-Fold蛋白质结构预测推理

配置数据库搜索与 `config/data.yaml`中的相关参数，下载已经训好的模型权重[MEGA_Fold_1.ckpt](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt)，运行以下命令启动推理。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --run_platform PLATFORM
            --input_path INPUT_FILE_PATH --checkpoint_path CHECKPOINT_PATH

选项：
--data_config        数据预处理参数配置
--model_config       模型超参配置
--input_path         输入文件目录，可包含多个.fasta/.pkl文件
--checkpoint_path    模型权重文件路径
--use_pkl            使用pkl数据作为输入，默认False
--run_platform       运行后端，Ascend或者GPU，默认Ascend
--mixed_precision    调用混合精度推理，默认0，全精度推理
```

对于多条序列推理，MEGA-Fold会基于所有序列的最长长度自动选择编译配置，避免重复编译。如需推理的序列较多，建议根据序列长度分类放入不同文件夹中分批推理。由于数据库搜索硬件要求较高，MEGA-Fold支持先做数据库搜索生成 `raw_feature`并保存为pkl文件，然后使用 `raw_feature`作为预测工具的输入，此时须将 `use_pkl`选项置为True，`examples`文件夹中提供了样例pkl文件与对应的真实结构，供测试运行，测试命令参考 `scripts/run_fold_infer_gpu.sh`。

推理结果保存在 `./result/` 目录下，每条序列的结果存储在独立文件夹中，以序列名称命名，文件夹中共有两个文件，pdb文件即为蛋白质结构预测结果，其中倒数第二列为氨基酸残基的预测置信度；timings文件保存了推理不同阶段时间信息以及推理结果整体的置信度。

```text
{"pre_process_time": 0.61, "model_time": 87.5, "pos_process_time": 0.02, "all_time ": 88.12, "confidence ": 93.5}
```

MEGA-Fold预测结果与真实结果对比：

7VGB_A，长度711，lDDT 92.3：

![7VGB_A](./images/7VGB_A.png)

### MEGA-Fold蛋白质结构预测训练

下载开源结构训练数据集[PSP dataset](http://ftp.cbi.pku.edu.cn/psp/)，使用以下命令启动训练：

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --is_training True
            --input_path INPUT_PATH --pdb_path PDB_PATH --run_platform PLATFORM

选项：
--data_config        数据预处理参数配置
--model_config       模型超参配置
--is_training        设置为训练模式
--input_path         训练输入数据（pkl文件，包含MSA与模板信息）路径
--pdb_path           训练标签数据（pdb文件，真实结构或知识蒸馏结构）路径
--run_platform       运行后端，Ascend或者GPU，默认Ascend
```

代码默认每50次迭代保存一次权重，权重保存在 `./ckpt`目录下。数据集下载及测试命令参考 `scripts/run_fold_train.sh`。

## 引用

[1] Jumper J, Evans R, Pritzel A, et al. Applying and improving AlphaFold at CASP14[J]. Proteins: Structure, Function, and Bioinformatics, 2021.

[2] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.

[3] Mirdita M, Ovchinnikov S, Steinegger M. ColabFold-Making protein folding accessible to all[J]. BioRxiv, 2021.

## 致谢

MEGA-Fold使用或参考了以下开源工具：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [HH Suite](https://github.com/soedinglab/hh-suite)
- [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
- [ML Collections](https://github.com/google/ml_collections)
- [NumPy](https://numpy.org)
- [OpenMM](https://github.com/openmm/openmm)

感谢这些开源工具所有的贡献者和维护者！
