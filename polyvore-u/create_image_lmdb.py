import os
import os.path

import lmdb
import torchutils
from tqdm import tqdm

output_dir = "processed"
input_dir = "release"

images = torchutils.files.scan_files(f"{input_dir}/images/291x291", suffix="jpg")
dst = f"{output_dir}/features/images"

os.makedirs(dst, exist_ok=True)
env = lmdb.open(dst, map_size=2 ** 40)
# open json file
with env.begin(write=True) as txn:
    for fn in tqdm(images):
        key = fn.split("/")[-1]
        with open(fn, "rb") as f:
            img_data = f.read()
            txn.put(key, img_data)
env.close()
