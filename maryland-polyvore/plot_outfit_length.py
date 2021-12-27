import json

import matplotlib.pyplot as plt
import numpy as np


def load_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


input_dir = "release"
train_outfits = load_json(f"{input_dir}/label/train_no_dup.json")
valid_outfits = load_json(f"{input_dir}/label/valid_no_dup.json")
test_outfits = load_json(f"{input_dir}/label/test_no_dup.json")
bins = np.arange(2, 10) + 0.5
plt.hist([len(outfit["items"]) for outfit in train_outfits], density=True, alpha=0.5, bins=bins, label="train")
plt.hist([len(outfit["items"]) for outfit in test_outfits], density=True, alpha=0.5, bins=bins, label="test")
plt.title("Histgram of outfit length ")
plt.xlabel("Number of items")
plt.ylabel("Density")
plt.legend()
plt.savefig("maryland_polyvore_outfit_length_hist.svg")
