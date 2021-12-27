#!/usr/bin/env python
import gzip
import io
import json
import os

import lmdb
import numpy as np
import torchutils
from tqdm import tqdm


def main():
    files = torchutils.files.scan_files("release/features")
    lmdb_dir = "processed/features/vgg"
    os.makedirs(lmdb_dir, exist_ok=True)
    with lmdb.open(lmdb_dir, map_size=2 ** 40) as env:
        with env.begin(write=True) as txn:
            for fn in tqdm(files, desc="Extract features"):
                item_id = fn.split("/")[-1].split(".")[0]
                with gzip.open(fn, "r") as f:
                    outfits = f.read()
                    feat = json.loads(outfits.decode("utf-8"))
                    feat = np.array(feat, np.float32)
                txn.put(item_id.encode(), feat)


if __name__ == "__main__":
    main()
