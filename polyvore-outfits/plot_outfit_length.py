import os

import matplotlib.pyplot as plt
import numpy as np
import torchutils

if __name__ == "__main__":
    input_dir = "release"
    files = ["train.json", "valid.json", "test.json"]
    nondisjoint = [
        len(outfit["items"])
        for fn in files
        for outfit in torchutils.io.load_json(os.path.join(input_dir, "nondisjoint", fn))
    ]
    disjoint = [
        len(outfit["items"])
        for fn in files
        for outfit in torchutils.io.load_json(os.path.join(input_dir, "disjoint", fn))
    ]

    bins = np.arange(2, 19) + 0.5
    plt.hist(nondisjoint, density=True, alpha=0.5, bins=bins, label="Polyvore Outfits")
    plt.hist(disjoint, density=True, alpha=0.5, bins=bins, label="Polyvore Outfits-D")
    plt.title("Histogram of outfit length ")
    plt.xlabel("Number of items")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("polyvore_outfits_length_hist.svg")
