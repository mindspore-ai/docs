# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/RELEASE.md)

## MindSpore 2.4.1 Release Notes

### 主要特性及增强

#### AutoParallel

- [STABLE] 支持split/concat分支通信计算并行，用户通过切分输入数据，形成可并行分支，分支间自动进行通信计算并行，降低通信开销。
- [STABLE] 支持Sequence pipeline，配套MindFormers的dev分支的LLama系列模型，通过引入Sequence维度拆分，降低流水线并行的Bubble以及显存开销。

#### PyNative

- [STABLE] PyNative模式通信算子默认按照通信域分配流，支持通信算子并发执行，协同并行策略优化，提供细粒度的通信掩盖，提升模型性能。

### 问题修复

- [IB0R4N](https://gitee.com/mindspore/mindspore/issues/IB0R4N)：修复在某些切分策略下，加载分布式权重精度不对的问题。

### 贡献者

bantao;caifubi;candanzg;chaijinwei;changzherui;chengbin;chujinjin;DeshiChen;dingjinshan;fary86;fuhouyu;gaoyong10;GuoZhibin;halo;haozhang;hedongdong;huangbingjian;hujiahui8;huoxinyou;jiangshanfeng;jiaorui;jiaxueyu;jshawjc;kisnwang;lichen;limingqi107;liubuyu;looop5;luochao60;luoyang;machenggui;MengXiangyu;Mrtutu;NaCN;panzhihui;qiuzhongya;shenhaojing;shilishan;tanghuikang;TuDouNi;wang_ziqi;weiyang;wujueying;XianglongZeng;xuxinglei;yang guodong;yanghaoran;yao_yf;yide12;yihangchen;YijieChen;YingtongHu;yuchaojie;YuJianfeng;zhangdanyang;ZhangZGC;zhengzuohe;zong_shuai;ZPaC;冯一航;胡彬;宦晓玲;李林杰;刘崇鸣;刘勇琪;任新;王禹程;王振邦;熊攀;俞涵;张栩浩;周一航;
