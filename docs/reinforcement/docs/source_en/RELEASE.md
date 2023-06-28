# MindSpore Reinforcement Release Notes

## MindSpore Reinforcement 0.3.0 Release Notes

### Major Features and Improvements

- [STABLE] Support DDPG reinforcement learning algorithm.

### API Change

#### Backwards Compatible Change

##### Python API

- Change the API of following classes: `Actor`, `Agent`. Their function names change to `act(self, phase, params)` and `get_action(self, phase, params)`. Moreover, some useless functions are deleted (`env_setter`, `act_init`, `evaluate`, `reset_collect_actor`, `reset_eval_actor, update` in `Actor`class, and `init`, `reset_all` in `Agent` class). Also the hierarchy relationship of configuration file changes. `ReplayBuffer`is moved out from the directory `actor`, and becomes a new key in `algorithm config`.
- Add the virtual base class of `Environment` class. It has `step`, `reset`functions and 5 `space` properties (`action_space`, `observation_space`, `reward_space`, `done_space` and `config`)

### Contributors

Thanks goes to these wonderful people:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

Contributions of any kind are welcome!
