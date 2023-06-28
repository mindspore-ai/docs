# MindSpore Reinforcement Release Notes

## MindSpore Reinforcement 0.3.0 Release Notes

### 主要特性和增强

- [STABLE] 支持DDPG强化学习算法

### 接口变更  

#### 后向兼容变更

##### Python接口

- 修改了`Actor`和`Agent`类的接口。它们的方法名被修改成`act(self, phase, params)`和`get_action(self, phase, params)`。除此之外，删除冗余方法(`Actor`类中的`env_setter`, `act_init`, `evaluate`, `reset_collect_actor`, `reset_eval_actor`, `update`, 和`Agent`类中的 `init`, `reset_all`)。修改配置文件中的层级结构，将`actor`目录下的`ReplayBuffer`移出作为`algorithm_config`中的一个单独键值。

- 增加了`Environment`类的虚基类。它提供`step`和`reset`方法以及5个`space`相关的属性(`action_space`, `observation_space`, `reward_space`, `done_space`和`config`)

### Contributors

感谢以下人员作出的贡献：

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

欢迎以任意形式对项目提供贡献!
