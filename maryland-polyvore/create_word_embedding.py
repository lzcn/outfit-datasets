#!/usr/bin/env python
import json
import pickle as pkl
import numpy as np
from tqdm.auto import tqdm


def load_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def main(input_dir, output_dir):
    train = load_json(f"{input_dir}/label/train_no_dup.json")
    valid = load_json(f"{input_dir}/label/valid_no_dup.json")
    tast = load_json(f"{input_dir}/label/test_no_dup.json")
    outfits = train + valid + tast
    words = dict()
    with open(f"{input_dir}/final_word_dict.txt") as f:
        for l in f.readlines():
            k, v = l.strip().split()
            words[k] = int(v)
    word_dict = dict()
    for w in words:
        word_dict[w] = len(word_dict)
    print("Number of words: {}, plus 1 unkown".format(len(word_dict)))

    item_words = dict()
    for outfit in tqdm(outfits, desc="Creating bag of words"):
        for item in outfit["items"]:
            item_id = item["image"].split("tid=")[-1]
            embd = [0] * (len(word_dict) + 1)
            for w in item["name"].split():
                if w in word_dict:
                    embd[word_dict[w]] += 1
                else:
                    embd[-1] += 1
            item_words[item_id] = np.array(embd)
    with open(f"{output_dir}/word_embedding.pkl", "wb") as f:
        pkl.dump(item_words, f)


if __name__ == "__main__":
    input_dir = "release"
    output_dir = "processed"
    main(input_dir, output_dir)
