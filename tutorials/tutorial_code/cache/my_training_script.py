import argparse
import mindspore.common.dtype as mstype
import mindspore.dataset as ds

parser = argparse.ArgumentParser(description='Cache Example')
parser.add_argument('--num_devices', type=int, default=1, help='Device num.')
parser.add_argument('--device', type=int, default=0, help='Device id.')
parser.add_argument('--session_id', type=int, default=1, help='Session id.')
args_opt = parser.parse_args()

schema = ds.Schema()
schema.add_column('image', de_type=mstype.uint8, shape=[2])
schema.add_column('label', de_type=mstype.uint8, shape=[1])

ds.config.set_seed(1)
ds.config.set_num_parallel_workers(1)

# apply cache to dataset
test_cache = ds.DatasetCache(session_id=args_opt.session_id, size=0, spilling=False)
dataset = ds.RandomDataset(schema=schema, total_rows=4, num_parallel_workers=1, cache=test_cache, num_shards=args_opt.num_devices, shard_id=args_opt.device)
num_iter = 0
for _ in dataset.create_dict_iterator():
    num_iter += 1
print("Got {} samples on device {}".format(num_iter, args_opt.device))
