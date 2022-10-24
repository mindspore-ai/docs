# Vertical Federated-Privacy Set Intersection

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/private_set_intersection.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Privacy Protection Background

With the rise in demand for digital transformation and the circulation of data elements, as well as the implementation of the Data Security Law, the Personal Information Protection Law and the EU General Data Protection Regulation (GDPR), privacy of data is increasingly becoming a necessary requirement in many scenarios. For example, when the dataset is sensitive information of users (medical diagnosis information, transaction records, identification codes, device unique identifier OAID, etc.) or secret information of the company, cryptography or desensitization must be used to ensure the confidentiality of the data before using it in the open state to achieve the goal of "usable but invisible" of the data in order to prevent information leakage. Considering two participants who jointly train a machine learning model (e.g., vertical federated learning) by using their respective data, the first step of this task is to align the sample sets of both parties, a process known as Entity Resolution. Traditional plaintext intersection inevitably reveals the OAID of the entire database and damages the data privacy of both parties, so the Privacy Set Intersection (PSI) technique is needed to accomplish this task.

PSI is a type of secure multi-party computing (MPC) protocol that takes data collection from two parties as input, after a series of hashing, encryption and data exchange steps, eventually outputs the intersection of the collection to an agreed output party, while ensuring that the participating parties cannot obtain any information about the data outside the intersection. The use of the PSI protocol in vertical federated learning tasks, in compliance with the GDPR requirement of Data Minimisation, i.e. there is no non-essential exposure of data, except for the parts necessary for the training process (intersections). From the data controller's perspective, the service has to share data appropriately, but wants to share only necessary data based on the service and not expose additional data to the public. It should be noted that while PSI can directly apply existing MPC protocols to its calculations, this often results in a large computational and communication overhead, which is not conducive to business. In this paper, we present a technique of combining Bloom Filter and Elliptic Curve with point multiplication Inverse Element offset to implement ECDH-PSI (Elliptic Curve Diffie-Hellman key Exchange-PSI) to better support cloud services and carry out privacy preserving set intersection computing services.

## Algorithm Process Introduction

The core idea of ECDH-PSI is that a piece of data is first encrypted by Alice and then encrypted by Bob, with the same result as exchanging the encryption order. One party sends the data encrypted with its own private key without revealing its privacy, and the other party re-encrypts it with its own private key based on the received encrypted data. If the encryption result is the same, the original data is the same.

The core optimization point of the inverse ECDH-PSI is to minimize the encryption computation based on the set of large amount of data when facing the scenario of intersection between two parties with unbalanced amount of data (Bob is the party with less data, $a$ and $b$ are the private keys of Alice and Bob respectively, the original data of both parties are mapped to the elliptic curve as $P_1$ and $P_2$ respectively, the point multiplication encryption of the elliptic curve with the private key $k$ is $P^k$ or $kP$, and the inverse of the private key $k$ is $k^{-1}$). Then after Alice executes $p_1^a$ and sends it to Bob, Bob no longer performs the encryption calculation based on it, but sends $p_2^b$ to Alice. After Alice sends $P_2^{ba}$, Bob completes the offset operation by point multiplying the inverse of its private key, i.e., calculating $P_2^{bab^{-1}}$ and comparing it with the $P_1^a$ sent by Alice. If the encryption result is the same, it means $P_1=P_2$. The flowchart of the inverse ECDH-PSI is shown in the figure, and the red letters indicate the received data from the other side.

![](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/images/inverse_ecdh_psi_flow.png)

The $bf$ in the figure stands for Bloom filter (bf). If you want to query whether an element exists in a collection, the basic method is to iterate through the collection to query, or sort the collection and use dichotomous lookup to query, but when the amount of data is too large, sorting does not support parallelism, which is very time-consuming. If a bloom filter is used, the elements of the set are mapped to a number of bits in an initial all-0 bit string by a number of hash functions, and all the elements of the sets share a single bit string. When querying, simply use the same number of hash functions to process the data to be queried, and directly access all the corresponding bits to see if they are activated to 1. If all of them are 1, it means that the data exists. Otherwise, it does not exist. The probability of collision can be achieved by controlling the number of hash functions. The communication overhead of the latter is lower compared to sending the entire set and sending a single bit string from the output of the Bloom filter. The computation can also be accelerated by parallelism during the creation of the bloom filter and the use of the filter for large-scale data queries.

## Quick Experience

### Front-end Needs

Finish installing the `mindspore-federated` library in the Python environment.

### Starting the Script

You can get the PSI start script for both sides from [MindSpore federated ST](https://gitee.com/mindspore/federated/blob/master/tests/st/psi/run_psi.py) and open two processes to simulate both sides. The start command of local device and local communication:

```python
python run_psi.py --comm_role="server" --http_server_address="127.0.0.1:8004" --remote_server_address="127.0.0.1:8005" --input_begin=1 --input_end=100

python run_psi.py --comm_role="client" --http_server_address="127.0.0.1:8005" --remote_server_address="127.0.0.1:8004" --input_begin=50 --input_end=150
```

- `input_begin` is used in conjunction with `input_end` to generate the dataset for intersection.
- `peer_input_begin` and `peer_input_end` indicate the start and end ranges of each other's data, taking `--need_check` as `True`, which can be intersected by the Python set1.intersection(set2) function to get the true result, and is used to check the correctness of the PSI.
- `---bucket_size` (optional) indicates the number of for loops that serially perform multiple bucket intersections.
- `--thread_num` (optional) indicates the number of parallel threads used for the calculation.
- To run plaintext intersection, add the parameter `--plain_intersection=True` to the command.

### Output Results

Before running the script, you can set the environment variable `export GLOG_v=1` to display the `INFO` level log, and also observe the operation of each phase within the protocol. After running the script, the intersection results will be printed out. As the amount of intersection data may be too large, the output here is limited to the first 20 intersection results.

```bash
PSI result: ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69'] (display limit: 20)
```

## Deep Experience

### Import Module

To run the privacy set intersection, you need to rely on the communication module and the intersection module of the Federated Library, which are imported as follows:

```python
from mindspore_federated.startup.vertical_federated_local import VerticalFederatedCommunicator, ServerConfig
from mindspore_federated._mindspore_federated import RunPSI
from mindspore_federated._mindspore_federated import PlainIntersection
```

### Data Preparation

Both `RunPSI` and `PlainIntersection` require input data in `List(String)` format, and methods for generating datasets via file reading and for loops are given here:

```python
def generate_input_data(input_begin_, input_end_, read_file_, file_name_):
    input_data_ = []
    if read_file_:
        with open(file_name_, 'r') as f:
            for line in f.readlines():
                input_data_.append(line.strip())
    else:
        input_data_ = [str(i) for i in range(input_begin_, input_end_)]
    return input_data_
```

The input parameters `input_begin_` and `input_end_` limit the data range of the for loop. `read_file_` and `file_name_` indicate whether to read the file and the path where the file is located. The file can be constructed by itself, each line representing one piece of data.

### Constructing Communication

Before calling this interface, a vertical federated communication instance needs to be initialized, as follows:

```python
http_server_config = ServerConfig(server_name=comm_role, server_address=http_server_address)
remote_server_config = ServerConfig(server_name=peer_comm_role, server_address=remote_server_address)
vertical_communicator = VerticalFederatedCommunicator(http_server_config=http_server_config,
                                                      remote_server_config=remote_server_config)
vertical_communicator.launch()
```

- `server_name` is determined by whether the process belongs to `server` or `client`. `comm_role` is assigned to the corresponding `server` or `client`, and `peer_comm_role_` indicates the role of the other party.
- The format of `server_address` is "IP:port". `http_server_address` is assigned to the `IP` and `port` information of the process, such as "127.0.0.1:8004". `remote_server_address` is assigned to the `IP` and `port` information of the other party.

### Starting Intersection

The external interfaces for secure set intersection are `RunPSI` and `PlainIntersection`, which are ciphertext and plaintext intersections respectively, with the same type and meaning of input and return results. Only ciphertext intersection `RunPSI` is described here:

```python
result = RunPSI(input_data, comm_role, peer_comm_role, bucket_id, thread_num)
```

- `input_data`: (list[string]), psi, the input data of one party.
- `comm_role`: (string), communication-related parameter, "server" or "client".
- `peer_comm_role`: (string), communication-related parameter, "server" or "client", different with comm_role.
- `bucket_id`: (int), outer part of the barrel, serial number of the pass-in barrel. `TypeError` error for passing in negative numbers, decimals or other types. If the value is different between two processes, the server will exit with an error and the client will block and wait.
- `thread_num`: (int), number of threads, natural number. 0 is the default value, which means use the maximum number of threads available on the machine minus 5, and other values will be limited to 1 to the maximum available on the machine. `TypeError` error for passing in negative numbers, decimals or other types.

### Output Results

The `result` is in `list[string]` format, which represents the intersection result and can be printed out by itself. Here's the method of the Python set intersection:

```python
def compute_right_result(self_input, peer_input):
    self_input_set = set(self_input)
    peer_input_set = set(peer_input)
    return self_input_set.intersection(peer_input_set)
```

The results of the above methods can be compared with `result` to check if they are consistent, and the correctness of the interface can be verified.
