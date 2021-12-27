#!/usr/bin/env python
import argparse
import os
import torchutils

import lmdb
from tqdm import tqdm


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Make Image LMDB", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--output-dit", default="processed/features/images", type=str, help="path for saving lmdb data.")
    parser.add_argument("--input-dir", default="release", type=str, help="path to the released data directory.")
    args = parser.parse_args()
    # fmt: on
    images = torchutils.files.scan_files(f"{args.input_dir}/images/291x291")
    os.makedirs(args.output_dit, exist_ok=True)
    env = lmdb.open(args.output_dit, map_size=2 ** 40)
    with env.begin(write=True) as txn:
        for fn in tqdm(images, desc="Processing images"):
            key = fn.split("/")[-1]
            with open(fn, "rb") as f:
                txn.put(key.encode("ascii"), f.read())
    env.close()
