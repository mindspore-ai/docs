# 分子体系建模

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindsponge/docs/source_zh_cn/xponge.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

MindSPONGE软件中包含一个轻量化且易于定制的分子体系建模工具，用于执行分子模拟的前处理和后处理。

## 分子体系建模工具能做什么

分子体系建模工具主要是为分子动力学软件MindSPONGE设计的，但是它可以输出一些通用格式的文件，如mol2和PDB，因此它也可以帮助其他分子建模程序。

## 如何获得

分子体系建模工具适用于所有操作系统(Windows/Linux/MacOS)。作为MindSPONGE的一部分，可以通过下载并安装MindSPONGE来获取。

## 安装

请参考[MindSPONGE安装教程](https://www.mindspore.cn/mindsponge/docs/zh-CN/master/intro.html#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B)。

| 依赖名称   | 使用说明                                        | 安装方式                 |
| :-------   | :--------------------------------------------- | :---------------------- |
| XpongeLib  | c/c++ compiled library for mindsponge.toolkits | `pip install XpongeLib` |
| pyscf      | quantum chemistry                              | `pip install pyscf`     |
| geometric  | geometry optimization                          | `pip install geometric` |
| rdkit      | cheminformatics                                | `conda install -c rdkit rdkit` |
| MDAnalysis | trajectory analysis                            | `pip install MDAnalysis` |

### 在Windows中无法使用的函数

- mindsponge.toolkits.assign.calculate_charge(method="RESP")
- mindsponge.toolkits.assign.resp
- mindsponge.toolkits.assign.resp.resp_fit

### 安装检查

通过运行以下指令检查`mindsponge.toolkits`是否成功调用。

```bash
python -m mindsponge.toolkits test --do base
```

## 力场

分子体系建模工具中包含多个力场，可以通过指令直接调用获取。

| 力场类型 | 分子类型 | 力场 | Module |
| :------ | :------- | :--- | :---- |
| AMBER | proteins | ff14SB | mindsponge.toolkits.forcefield.amber.ff14sb |
| | proteins | ff19SB | mindsponge.toolkits.forcefield.amber.ff19sb |
| | lipids | lipid14 | mindsponge.toolkits.forcefield.amber.lipid14 |
| | lipids | lipid17 | mindsponge.toolkits.forcefield.amber.lipid17 |
| | carbohydrates (D-Pyranose) | glycam-06j | mindsponge.toolkits.forcefield.amber.glycam-06j |
| | carbohydrates (L-Pyranose) | glycam-06j | mindsponge.toolkits.forcefield.amber.glycam-06j |
| | carbohydrates (D-Furanose) | glycam-06j | mindsponge.toolkits.forcefield.amber.glycam-06j |
| | carbohydrates (L-Furanose) | glycam-06j | mindsponge.toolkits.forcefield.amber.glycam-06j |
| | glycoprotein | glycam-06j | mindsponge.toolkits.forcefield.amber.glycam-06j |
| | organic molecules | gaff | mindsponge.toolkits.forcefield.amber.gaff |
| | water/ions | tip3p | mindsponge.toolkits.forcefield.amber.tip3p |
| | water/ions | spce | mindsponge.toolkits.forcefield.amber.spce |
| | water/ions | tip4pew | mindsponge.toolkits.forcefield.amber.tip4pew |
| | water/ions | opc | mindsponge.toolkits.forcefield.amber.opc |
| CHARMM27 | proteins | protein | mindsponge.toolkits.forcefield.charmm27.protein |
| | water/ions | tip3p | mindsponge.toolkits.forcefield.charmm27.tip3p |
| | water/ions | tip3p | mindsponge.toolkits.forcefield.charmm27.tip3p |
