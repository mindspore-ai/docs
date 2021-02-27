# Function Differences of init

PyTorch: This interface supports three kinds of collective communication: MPI, Gloo, and NCCL. It initializes backend and also provides configuration, such as `world_size`, `rank`, `timeout`, etc.

MindSpore：This interface currently supports only two kinds of collective communication: HCCL and NCCL. The configuration of `world_size`, `rank` and `timeout` is not set in this interface. The corresponding environment variable needs to be set before calling this interface.
