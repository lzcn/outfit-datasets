#!/usr/bin/env python
# %%
# create dataset for NGNN

import os

import numpy as np
import torchutils
from sklearn.metrics import roc_auc_score

# %%
# load polyvore maryland dataset

inputDir = "maryland-polyvore/release"
outputDir = "maryland-polyvore/processed"
dataDir = "maryland-polyvore/ngnn"

trainOutfits = torchutils.io.load_json(f"{inputDir}/label/train_no_dup.json")
validOutfits = torchutils.io.load_json(f"{inputDir}/label/valid_no_dup.json")
testOutfits = torchutils.io.load_json(f"{inputDir}/label/test_no_dup.json")
allOutfits = trainOutfits + validOutfits + testOutfits

print(
    "Number of outfits: {} = {} + {} + {}".format(
        len(allOutfits), len(trainOutfits), len(validOutfits), len(testOutfits)
    )
)


# %%

allItems = set()
itemDict = dict()
num_items = 0
cateIds = set()
outfitSet = set()

min_size = np.inf
max_size = -np.inf
for outfit in allOutfits:
    size = len(outfit["items"])
    min_size = min(size, min_size)
    max_size = max(size, max_size)
    set_id = outfit["set_id"]
    outfitSet.add(set_id)
    num_items += size
    for item in outfit["items"]:
        category = item["categoryid"]
        cateIds.add(category)
        # the unique id of item
        tid = item["image"].split("tid=")[-1]
        if tid not in itemDict:
            itemDict[tid] = []
        itemDict[tid].append(os.path.join(f"{inputDir}/images", "{}/{}.jpg".format(set_id, item["index"])))
        allItems.add(item["image"])
print("Number of categories: {}".format(len(cateIds)))
print("Number of unique items: {:,}".format(len(allItems)))
print("Reuse ratio: {:.3f} = {:,} / {:,}".format(num_items / len(allItems), num_items, len(allItems)))
print("Average number of items in an outfit: {:.2f}".format(num_items / len(allOutfits)))
print("Minimum number of items in an outfit: {}".format(min_size))
print("Maximum number of items in an outfit: {}".format(max_size))

# %% [markdown]
# ## Data processing for NGNN
#
# NGNN use the training outfit to clean the dataset.

# %%


def get_valid_category(outfits, num=100):
    used_cate = dict()
    all_cate = dict()
    for outfit in outfits:
        for item in outfit["items"]:
            category = item["categoryid"]
            all_cate[category] = all_cate.get(category, 0) + 1
    for k, s in all_cate.items():
        if s > num:
            used_cate[k] = s
    return used_cate


validCate = get_valid_category(trainOutfits, 100)
print("Number of valid categories: {:,}".format(len(validCate)))
cateMap = dict()
for i, cate in enumerate(sorted(list(validCate.keys()))):
    cateMap[cate] = i

# check category set

cate_idx_map = torchutils.io.load_json("maryland-polyvore/ngnn/cid2rcid_100.json")
cate_idx_map = {int(k): v for k, v in cate_idx_map.items()}
assert cateMap == cate_idx_map

# %%
cateItems = [set() for _ in validCate.keys()]
for outfit in allOutfits:
    for item in outfit["items"]:
        category = item["categoryid"]
        if category in validCate:
            item_id = item["image"].split("tid=")[-1]
            cateItems[cateMap[category]].add(item_id)
cateItems = [sorted(list(v)) for v in cateItems]
itemIndex = [{key: i for i, key in enumerate(items)} for items in cateItems]
torchutils.io.save_json(f"{outputDir}/ngnn/items.json", cateItems)

for c, items in enumerate(cateItems):
    for i in range(len(items)):
        assert itemIndex[c][cateItems[c][i]] == i
# %% [markdown]
# ## Clean each split

# %%
# {set_id}_{index}: {item_id}
index2ItemId = dict()
# {item_id}: {item_cate}
itemCate = dict()
for outfit in allOutfits:
    set_id = outfit["set_id"]
    for item in outfit["items"]:
        tid = item["image"].split("tid=")[-1]
        index2ItemId["{}_{}".format(set_id, item["index"])] = tid
        itemCate[tid] = item["categoryid"]


# %%
def clean_outfits(outfits, valid_category):
    """Clean outfits.

    1. items with valid category
    2. no duplicated category in an outfit
    3. number of items > 2
    """
    cleaned_outfits = dict()
    for outfit in outfits:
        set_id = outfit["set_id"]
        cleaned = {
            "items_index": [],
            "items_category": [],
            "items_id": [],
        }
        for item in outfit["items"]:
            item_cate = item["categoryid"]
            if item_cate in valid_category and item_cate not in cleaned["items_category"]:
                cleaned["items_category"].append(item_cate)
                cleaned["items_index"].append(item["index"])
                cleaned["items_id"].append(item["image"].split("tid=")[-1])
        if len(cleaned["items_index"]) > 2:
            cleaned_outfits[set_id] = cleaned
    return cleaned_outfits


# %%
# cleaned data:
#        set_id: {"items_index": [], "items_category": [], "items_id": []}
# original data:
#        set_id: {"items_index": [], "items_category": [], "set_id": set_id}

train_outfits = torchutils.io.load_json("maryland-polyvore/ngnn/train_no_dup_new_100.json")
train_outfits = {outfit["set_id"]: outfit for outfit in train_outfits}
test_outfits = torchutils.io.load_json("maryland-polyvore/ngnn/test_no_dup_new_100.json")
test_outfits = {outfit["set_id"]: outfit for outfit in test_outfits}

cleanTrainOutfits = clean_outfits(trainOutfits, validCate)
cleanValidOutfits = clean_outfits(validOutfits, validCate)
cleanTestOutfits = clean_outfits(testOutfits, validCate)

# check data
for key, value in cleanTestOutfits.items():
    assert value["items_index"] == test_outfits[key]["items_index"]
for key, value in cleanTrainOutfits.items():
    assert value["items_index"] == train_outfits[key]["items_index"]

# show size
print(len(cleanTrainOutfits), len(cleanValidOutfits), len(cleanTestOutfits))


# %%
def count_frequency(outfits):
    cate_freq = dict()
    for outfit in outfits.values():
        for cate in outfit["items_category"]:
            cate_freq[cate] = cate_freq.get(cate, 0) + 1
    return cate_freq


cate_freq = count_frequency(cleanTrainOutfits)

# check frequency
cate_idx_map = torchutils.io.load_json("maryland-polyvore/ngnn/cid2rcid_100.json")
cate_summary = torchutils.io.load_json("maryland-polyvore/ngnn/category_summarize_100.json")
original_cate = {x["id"]: x["frequency"] for x in cate_summary}
for key, freq in original_cate.items():
    assert cate_freq[key] == freq, (cate_freq[key], freq)

print("Minimum frequency: {:,}".format(min(cate_freq.values())))
print("Maximum frequency: {:,}".format(max(cate_freq.values())))


# %%
# get negative outfits
with open(f"{inputDir}/label/fashion_compatibility_prediction.txt") as f:
    lines = f.readlines()

cleanTestOutfitsNeg = []
for line in lines:
    label, *items = line.strip().split(" ")
    if int(label) == 0:
        set_id = items[0].split("_")[0]
        outfit = {"items_category": [], "items_id": []}
        for item in items:
            item_id = index2ItemId[item]
            item_category = itemCate[item_id]
            if False or item_category in validCate and item_category not in outfit["items_category"]:
                outfit["items_category"].append(item_category)
                outfit["items_id"].append(item_id)
        if False or len(outfit["items_category"]) > 2:
            cleanTestOutfitsNeg.append(outfit)

print("Number of negative outfits: {}".format(len(cleanTestOutfitsNeg)))


# %%
num_nodes = len(validCate)

co_occur = np.zeros((num_nodes, num_nodes), dtype=np.int64)
frequency = np.zeros((1, num_nodes), dtype=np.int64)

for outfit in cleanTrainOutfits.values():
    cate = [cateMap[i] for i in outfit["items_category"]]
    size = len(cate)
    for i in range(size):
        frequency[0, cate[i]] += 1
        for j in range(i + 1, size):
            co_occur[cate[i], cate[j]] += 1
            co_occur[cate[j], cate[i]] += 1

#    c_{ij} / c_j
# -------------------
# \sum c_{ik} / c_k
# torchutils.io.save_csv("processed/ngnn/frequency.txt", frequency)
# torchutils.io.save_csv("processed/ngnn/cooccurrence.txt", co_occur)
fashionGraph = (co_occur / frequency) / (co_occur / frequency).sum(axis=1, keepdims=True)
print((fashionGraph > 0).sum())
print(fashionGraph.shape)


# %%
# convert the json data into tuples
MAX_SIZE = 8


def convert(outfits):
    tuples = []
    for outfit in outfits:
        items = outfit["items_id"]
        cates = [cateMap[c] for c in outfit["items_category"]]
        items = [itemIndex[c][i] for c, i in zip(cates, items)]
        size = len(cates)
        append = [-1] * (MAX_SIZE - size)
        # user_id, size, [items], [cates]
        tuples.append([0] + [size] + items + append + cates + append)
    return np.array(tuples).astype(np.int64)


# %%
train_pos = convert(cleanTrainOutfits.values())
valid_pos = convert(cleanValidOutfits.values())
test_pos = convert(cleanTestOutfits.values())
test_neg = convert(cleanTestOutfitsNeg)


# %%
torchutils.io.save_csv(f"{outputDir}/ngnn/train_pos", train_pos)
torchutils.io.save_csv(f"{outputDir}/ngnn/valid_pos", valid_pos)
torchutils.io.save_csv(f"{outputDir}/ngnn/test_pos", test_pos)
torchutils.io.save_csv(f"{outputDir}/ngnn/test_neg", test_neg)

# %%

tuples = np.vstack((test_pos, test_neg))
y_true = [1] * len(test_pos) + [0] * len(test_neg)
y_pred = []
for tpl in tuples:
    size = tpl[1]
    cate = tpl[MAX_SIZE + 2 :]
    scores = []
    for i in range(size):
        for j in range(i + 1, size):
            scores.append(fashionGraph[cate[i], cate[j]])
            scores.append(fashionGraph[cate[j], cate[i]])
    y_pred.append(np.mean(scores))

print("AUC using fashion graph: {:.3f}".format(roc_auc_score(y_true, y_pred)))

# %%
