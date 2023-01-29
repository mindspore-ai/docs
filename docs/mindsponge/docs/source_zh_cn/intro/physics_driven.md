# 物理驱动

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_zh_cn/intro/physics_driven.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

传统的分子动力学模拟主要利用物理知识对分子体系进行计算模拟。

通常，研究体系会被设定为由许多质点组成，每个质点代表一个原子（全原子模拟）或若干原子（粗粒化模拟）。这些质点彼此之间存在一定的相互作用势能，根据理论力学即可求解运动方程，获得运动轨迹，以此可以进行动力学研究。

根据统计力学的系综理论，在模拟时间足够长的情况下，得到的分子轨迹中出现的每一个构象都在对应的系综里等概率分布，以此可以进行热力学研究。

分子动力学中质点之间存在的相互作用势能被称为力场。力场既决定了分子的运动，也决定了系统的系综。

构建合适的真实力场，就可以模拟真实物理场景下的微观世界；而添加人为设计的偏置势，则可以获得由物理驱动的分子采样结果。
