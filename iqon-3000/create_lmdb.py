#!/usr/bin/env python
import argparse
import os

import torchutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create LMDB for all images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lmdb-dir", default="processed/features/images_m")
    parser.add_argument("--image-dir", default="raw/images")
    parser.add_argument("--image-size", default="middle")
    args = parser.parse_args()
    if args.image_size == "small":
        suffixs = ("_s.jpg",)
    elif args.image_size == "middle":
        suffixs = ("_m.jpg",)
    elif args.image_size == "large":
        suffixs = ("_l.jpg",)
    else:
        raise KeyError
    image_list = torchutils.files.scan_files(args.image_dir, suffixs, recursive=True)
    src = {os.path.basename(path).split("_")[0]: path for path in image_list}
    torchutils.data.create_lmdb(args.lmdb_dir, src)
