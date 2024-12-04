# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_en/RELEASE.md)

## MindSpore 2.4.1 Release Notes

### Major Features and Improvements

#### AutoParallel

- [STABLE] Split/concat branch communication computation parallel is supported. Users split input data to form parallelizable branches. Automatic communication computing parallelism is performed between branches, reducing communication overhead.
- [STABLE] Sequence pipelines are supported. The LLama series models for the dev branch of MindFormers reduces the Bubble as well as the memory overhead of pipeline parallelism by introducing Sequence dimension splitting.

#### PyNative

- [STABLE] In PyNative mode, communication operators are assigned streams by default based on the communication domain. They support concurrent execution of communication operators, optimize collaborative parallel strategies, provide fine-grained communication masking, and enhance model performance.

### Bug Fixes

- [IB0R4N](https://gitee.com/mindspore/mindspore/issues/IB0R4N): Fixed the problem of loading distributed weights with inaccurate accuracy under certain splitting strategies.

### Contributors

bantao;caifubi;candanzg;chaijinwei;changzherui;chengbin;chujinjin;DeshiChen;dingjinshan;fary86;fuhouyu;gaoyong10;GuoZhibin;halo;haozhang;hedongdong;huangbingjian;hujiahui8;huoxinyou;jiangshanfeng;jiaorui;jiaxueyu;jshawjc;kisnwang;lichen;limingqi107;liubuyu;looop5;luochao60;luoyang;machenggui;MengXiangyu;Mrtutu;NaCN;panzhihui;qiuzhongya;shenhaojing;shilishan;tanghuikang;TuDouNi;wang_ziqi;weiyang;wujueying;XianglongZeng;xuxinglei;yang guodong;yanghaoran;yao_yf;yide12;yihangchen;YijieChen;YingtongHu;yuchaojie;YuJianfeng;zhangdanyang;ZhangZGC;zhengzuohe;zong_shuai;ZPaC;冯一航;胡彬;宦晓玲;李林杰;刘崇鸣;刘勇琪;任新;王禹程;王振邦;熊攀;俞涵;张栩浩;周一航;
