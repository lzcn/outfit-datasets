#!/usr/bin/env python
import argparse
import json
import os

import torchutils
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(
    description="Merge all JSON file", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--data-dir", default="raw/images")
parser.add_argument("--output", default="processed/outfits.json")
args = parser.parse_args()

data_dir = args.data_dir
outfit_file = args.output

if os.path.exists(outfit_file):
    print(f"File {outfit_file} exits.")
    exit(0)
label_dir = os.path.dirname(outfit_file)

if not os.path.exists(label_dir):
    os.mkdir(label_dir)

json_files = torchutils.files.scan_files(data_dir, "json", recursive=True)
num_files = len(json_files)
print("Number of json files: {}".format(num_files))

outfits = []
for fn in tqdm(json_files):
    with open(fn, "r") as f:
        try:
            info = json.load(f)
        except json.JSONDecodeError as e:
            tqdm.write("Error {} with file {}".format(e, fn))
            info = dict()
        if len(info):
            outfits.append(info)
with open(outfit_file, "w") as f:
    json.dump(outfits, f)
print("Number of outfits after cleaning: {}".format(len(outfits)))
