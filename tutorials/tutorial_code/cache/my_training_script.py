import argparse
import mindspore.dataset as ds

parser = argparse.ArgumentParser(description='Cache Example')
parser.add_argument('--num_devices', type=int, default=1, help='Device num.')
parser.add_argument('--device', type=int, default=0, help='Device id.')
parser.add_argument('--session_id', type=int, default=1, help='Session id.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

# apply cache to dataset
test_cache = ds.DatasetCache(session_id=args_opt.session_id, size=0, spilling=False)
dataset = ds.Cifar10Dataset(dataset_dir=args_opt.dataset_path, num_samples=4, shuffle=False, num_parallel_workers=1,
                            num_shards=args_opt.num_devices, shard_id=args_opt.device, cache=test_cache)
num_iter = 0
for _ in dataset.create_dict_iterator():
    num_iter += 1
print("Got {} samples on device {}".format(num_iter, args_opt.device))
