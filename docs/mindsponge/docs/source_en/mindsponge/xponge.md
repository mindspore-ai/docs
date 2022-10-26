# Molecular System Modeling

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindscience/docs/source_en/mindsponge/xponge.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

MindSPONGE software includes a lightweight and easy to customize molecular system modeling tool, which is used to perform pre-processing and post-processing of molecular simulation.

## What can molecular system modeling tools do

The molecular system modeling tool is mainly designed for the molecular dynamics software MindSPONGE, but it can output some files in common formats, such as mol2 and PDB, so it can also help other molecular modeling programs.

## How to get

Molecular system modeling tools are applicable to all operating systems (Windows/Linux/MacOS). As a part of MindSPONGE, it can be obtained by downloading and installing MindSPONGE.

## Install

Please refer to [MindSPONGE Installation Tutorial](https://www.mindspore.cn/mindscience/docs/en/master/mindsponge/intro.html) .

| Dependent Name   | instructions                                        | Installation method                 |
| :-------   | :--------------------------------------------- | :---------------------- |
| XpongeLib  | c/c++ compiled library for mindsponge.toolkits | `pip install XpongeLib` |
| pyscf      | quantum chemistry                              | `pip install pyscf`     |
| geometric  | geometry optimization                          | `pip install geometric` |
| rdkit      | cheminformatics                                | `conda install -c rdkit rdkit` |
| MDAnalysis | trajectory analysis                            | `pip install MDAnalysis` |

### Functions that cannot be used in Windows

- mindsponge.toolkits.assign.calculate_charge(method="RESP")
- mindsponge.toolkits.assign.resp
- mindsponge.toolkits.assign.resp.resp_fit

### Installation inspection

Run the following command to check whether 'mindspring. toolkits' is called successfully.

```bash
python -m mindsponge.toolkits test --do base
```

## Force field

The molecular system modeling tool contains multiple positions, which can be directly called through instructions.

| Force field type | Molecular type | force field | Module |
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
