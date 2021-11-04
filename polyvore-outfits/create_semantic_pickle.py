#!/usr/bin/env python
import argparse
import os
import pickle as pkl

import numpy as np
import torchutils
from tqdm import tqdm


def extract_semantic(data_set, data_dir, output_dir):
    save_dir = os.path.join(output_dir, data_set)
    os.makedirs(save_dir, exist_ok=True)
    metaData = torchutils.io.load_json(f"{data_dir}/polyvore_item_metadata.json")
    with open(f"{data_dir}/{data_set}/train_hglmm_pca6000.txt", "r") as f:
        lines = f.readlines()
    print(f"Number of lines: {len(lines)}")

    semantic_features = dict()
    for line in tqdm(lines, desc=f"Extracting semantic features for {data_set}"):
        split = line.strip().split(",")
        label, data = split[:-6000], split[-6000:]
        label = ",".join(label)
        data = np.array(list(map(float, data))).astype(np.float32)
        semantic_features[label] = data

    features = {}
    for key, value in metaData.items():
        desc = value["title"] if value["title"] else value["url_name"]
        desc = desc.replace("\n", "").strip().lower()
        if desc in semantic_features:
            features[key] = semantic_features[desc]
    print("Number of semantic items:", len(features))

    with open(f"{save_dir}/semantic.pkl", "wb") as f:
        pkl.dump(features, f)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Create semantic features", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--output-dir", default="processed", type=str, help="path for saving semantic data.")
    parser.add_argument("--input-dir", default="release", type=str, help="path to the released data directory.")
    args = parser.parse_args()
    # fmt: on
    data_set = "nondisjoint"
    extract_semantic(data_set, args.input_dir, args.output_dir)
    data_set = "disjoint"
    extract_semantic(data_set, args.input_dir, args.output_dir)
