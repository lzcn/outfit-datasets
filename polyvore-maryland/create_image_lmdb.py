#!/usr/bin/env python
import json
import os

import lmdb
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def load_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def get_images(input_dir):
    train = load_json(f"{input_dir}/label/train_no_dup.json")
    valid = load_json(f"{input_dir}/label/valid_no_dup.json")
    tast = load_json(f"{input_dir}/label/test_no_dup.json")
    outfits = train + valid + tast

    item_images = dict()

    for outfit in outfits:
        set_id = outfit["set_id"]
        for item in outfit["items"]:
            index = item["index"]
            item_id = item["image"].split("tid=")[-1]
            if item_id not in item_images:
                item_images[item_id] = []
            item_images[item_id].append(os.path.join(f"{input_dir}/images", "{}/{}.jpg".format(set_id, index)))

    return item_images


def check_reuse_images(item_images):
    for k, v in tqdm(item_images.items()):
        if len(v) > 1:
            images = []
            for fn in v:
                with open(fn, "rb") as f:
                    images.append(np.array(Image.open(f).convert("RGB")))
            item_images[k] = images
    error = dict()
    for k, v in tqdm(item_images.items(), desc="Checking images"):
        imgs = np.stack(v)
        mean = (imgs - imgs.mean(axis=0)).mean()
        error[k] = mean
    print("Mean error: {:.3f}".format(np.array(list(error.values())).mean()))


def save_as_lmdb(item_images, output_dir):
    dst = f"{output_dir}/features/images"
    env = lmdb.open(dst, map_size=2 ** 40)
    # open json file
    with env.begin(write=True) as txn:
        for item_id, item_path in tqdm(item_images.items(), desc="Writing images"):
            fn = item_path[0]
            with open(fn, "rb") as f:
                img_data = f.read()
                txn.put(item_id.encode("ascii"), img_data)
    env.close()


if __name__ == "__main__":
    input_dir = "release"
    output_dir = "processed"
    item_images = get_images(input_dir)
    check_reuse_images(item_images)
    save_as_lmdb(item_images, output_dir)
