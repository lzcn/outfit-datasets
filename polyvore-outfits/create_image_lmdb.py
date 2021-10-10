import os
import os.path

import lmdb
import torchutils
from tqdm import tqdm

output_dir = "processed"
input_dir = "release"

meta_data = torchutils.io.load_json(f"{input_dir}/polyvore_item_metadata.json")
dst = f"{output_dir}/features/images"

os.makedirs(dst, exist_ok=True)
env = lmdb.open(dst, map_size=2 ** 40)
# open json file
with env.begin(write=True) as txn:
    for item_id in tqdm(meta_data.keys()):
        key = item_id.encode("ascii")
        fn = os.path.join(f"{input_dir}/images/", "{}.jpg".format(item_id))
        with open(fn, "rb") as f:
            img_data = f.read()
            txn.put(key, img_data)
env.close()
