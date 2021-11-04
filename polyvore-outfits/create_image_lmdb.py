#!/usr/bin/env python
import argparse
import json
import os
import torchutils

import lmdb
from tqdm import tqdm


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Make LMDB", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--output-dit", default="processed/features/images", type=str, help="path for saving lmdb data.")
    parser.add_argument("--input-dir", default="release", type=str, help="path to the released data directory.")
    args = parser.parse_args()
    # fmt: on
    meta_data = torchutils.io.load_json(f"{args.input_dir}/polyvore_item_metadata.json")
    os.makedirs(args.output_dit, exist_ok=True)
    env = lmdb.open(args.output_dit, map_size=2 ** 40)
    with env.begin(write=True) as txn:
        for item_id in tqdm(meta_data.keys(), desc="Processing images"):
            fn = os.path.join(f"{args.input_dir}/images/", "{}.jpg".format(item_id))
            with open(fn, "rb") as f:
                txn.put(item_id.encode("ascii"), f.read())
    env.close()
