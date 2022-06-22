#!/usr/bin/env python
# %%
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchutils

# %% data configuration
inputDir = "release"
outputDir = "processed"
# dataSet = "nondisjoint"
dataSet = "disjoint"
minSize = 4
maxSize = 8
dataDir = os.path.join(inputDir, dataSet)
saveDir = os.path.join(outputDir, dataSet + "-0408")

os.makedirs(saveDir, exist_ok=True)

# %% [markdown]
# ## Example item format
# ```json
# {
#     "category_id": "29",
#     "catgeories": ["Women's Fashion", "Clothing", "Shorts"],
#     "description": "Blue High Waist Plain Polyester Loose Button Fly Street "
#     "Shorts, Size Available: S,M,L,XL S, M, L, XL Plain Blue "
#     "Polyester Style: Street.",
#     "related": [
#         "High waisted jean shorts",
#         "High-waisted denim shorts",
#         "Blue shorts",
#         "High-rise shorts",
#         "Highwaist shorts",
#         "Button shorts",
#     ],
#     "semantic_category": "bottoms",
#     "title": "Blue High Waist Buttons Denim Shorts",
#     "url_name": "blue high waist buttons denim",
# }
# ```

# %%

# item meta data
metaData = torchutils.io.load_json(f"{inputDir}/polyvore_item_metadata.json")

# item id to item semantic category
itemType = {}

# fine grained categories to semantic category
fineGrained2Semantic = dict()

# set of all semantic categories
semanticSet = set()
for k, v in metaData.items():
    semanticSet.add(v["semantic_category"])
    fineGrained2Semantic[v["category_id"]] = v["semantic_category"]
    itemType[k] = v["semantic_category"]

# smenatic category to index id
semanticDict = dict()
for cate in sorted(semanticSet):
    semanticDict[cate] = len(semanticDict)

print("Number of items: {:,}".format(len(metaData)))
print("Number of fine-grained categories: {:,}".format(len(fineGrained2Semantic)))
print("Number of semantic categories: {:,}".format(len(semanticDict)))

print("{:=^30}".format(" Semantic category "))
print(torchutils.format_display(semanticDict))

# %% Reformat the outfit tuples

with open(f"{inputDir}/{dataSet}/typespaces.p", "rb") as f:
    types = pkl.load(f)

print("Number of conditions (original):", len(types))
data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %% clean outfit with given size


def clean_outfits(outfits, min_size=-np.inf, max_size=np.inf):
    """Clean outfits with size between min_size and max_size"""
    results = []
    for outfit in outfits:
        items = outfit["items"]
        size = len(items)
        if min_size <= size <= max_size:
            results.append(outfit)
    return results


def infer_min_max_size(outfits):
    min_size = np.inf
    max_size = -np.inf
    for outfit in outfits:
        size = len(outfit["items"])
        if size < min_size:
            min_size = size
        if size > max_size:
            max_size = size
    return min_size, max_size


# %% before cleaning
trainOutfits = torchutils.io.load_json(os.path.join(dataDir, "train.json"))
validOutfits = torchutils.io.load_json(os.path.join(dataDir, "valid.json"))
testOutfits = torchutils.io.load_json(os.path.join(dataDir, "test.json"))
print("Example outfit: \n{}".format(torchutils.format_display(np.random.choice(trainOutfits))))

print(
    "[B] Number of {} outfits: {:,} = {:,}(train) + {:,}(valid) + {:,}(test)".format(
        dataSet,
        len(trainOutfits) + len(validOutfits) + len(testOutfits),
        len(trainOutfits),
        len(validOutfits),
        len(testOutfits),
    )
)

# %% after cleaning
trainOutfits = clean_outfits(trainOutfits, minSize, maxSize)
validOutfits = clean_outfits(validOutfits, minSize, maxSize)
testOutfits = clean_outfits(testOutfits, minSize, maxSize)

allOutfits = trainOutfits + validOutfits + testOutfits

print(
    "[A] Number of {} outfits: {:,} = {:,}(train) + {:,}(valid) + {:,}(test)".format(
        dataSet,
        len(trainOutfits) + len(validOutfits) + len(testOutfits),
        len(trainOutfits),
        len(validOutfits),
        len(testOutfits),
    )
)

MIN_SIZE, MAX_SIZE = infer_min_max_size(allOutfits)
MIN_TRAIN_SIZE, MAX_TRAIN_SIZE = infer_min_max_size(trainOutfits)
MIN_VALID_SIZE, MAX_VALID_SIZE = infer_min_max_size(validOutfits)
MIN_TEST_SIZE, MAX_TEST_SIZE = infer_min_max_size(testOutfits)
print("Outfit size: {:,} - {:,}".format(MIN_SIZE, MAX_SIZE))
print("Train size: {:,} - {:,}".format(MIN_TRAIN_SIZE, MAX_TRAIN_SIZE))
print("Valid size: {:,} - {:,}".format(MIN_VALID_SIZE, MAX_VALID_SIZE))
print("Test size: {:,} - {:,}".format(MIN_TEST_SIZE, MAX_TEST_SIZE))

# %%
# create item list
itemSet = [set() for _ in range(len(semanticSet))]

for outfit in allOutfits:
    items = outfit["items"]
    for item in outfit["items"]:
        item_id = item["item_id"]
        item_type = itemType[item_id]
        type_id = semanticDict[item_type]
        itemSet[type_id].add(item_id)

# sort items to get consistent order of each run
itemList = [sorted(list(items)) for items in itemSet]
itemIndex = [{key: index for index, key in enumerate(items)} for items in itemList]
print("Number of items in each category: ", list(map(len, itemList)))
torchutils.io.save_json(f"{saveDir}/items.json", itemList)


# %%


def count_items(outfits):
    sizes = [len(o["items"]) for o in outfits]
    return sizes


def count_conditions(outfits):
    conditions = set()
    for outfit in outfits:
        items = outfit["items"]
        size = len(items)
        for i in range(size):
            for j in range(size):
                id_i = items[i]["item_id"]
                id_j = items[j]["item_id"]
                if (itemType[id_i], itemType[id_j]) in conditions:
                    continue
                if (itemType[id_j], itemType[id_i]) in conditions:
                    continue
                conditions.add((itemType[id_i], itemType[id_j]))
    return conditions


# %%
def show_statistic(outfits, phase="train"):
    min_size, max_size = infer_min_max_size(outfits)
    plt.hist(count_items(outfits), bins=np.arange(min_size - 1, max_size + 1) + 0.5)
    plt.title("Histogram of outfits ({})".format(phase))
    num_conditions = count_conditions(outfits)
    print("Min size: {}".format(min_size))
    print("Max size: {}".format(max_size))
    print("Number conditions ({}): {}".format(phase, len(num_conditions)))
    data = [[i, j, "+"] for i, j in (types)]
    df = pd.DataFrame(data)
    return df.pivot(index=0, columns=1, values=2)


# %%

print("Number of items ({}):".format(dataSet))
print(torchutils.format_display({k: v for k, v in enumerate(np.bincount(count_items(allOutfits)))}))
show_statistic(allOutfits, "all")

# %%
show_statistic(trainOutfits, "train")
# %%
show_statistic(validOutfits, "valid")
# %%
show_statistic(testOutfits, "test")


# %%
def convert_tuples(outfits, max_size):
    tuples = []
    for outfit in outfits:
        items = outfit["items"]
        size = len(items)
        m = max_size - size
        item_ids = [i["item_id"] for i in items]
        types = [semanticDict[itemType[i]] for i in item_ids]
        items = [itemIndex[c][i] for c, i in zip(types, item_ids)]
        tuples.append([0, size] + items + [-1] * m + types + [-1] * m)
    return np.array(tuples)


# %%
trainTuples = convert_tuples(trainOutfits, MAX_TRAIN_SIZE)
validTuples = convert_tuples(validOutfits, MAX_VALID_SIZE)
testTuples = convert_tuples(testOutfits, MAX_TEST_SIZE)
torchutils.io.save_csv(f"{saveDir}/train_pos", trainTuples)
torchutils.io.save_csv(f"{saveDir}/valid_pos", validTuples)
torchutils.io.save_csv(f"{saveDir}/test_pos", testTuples)

# %%
name2Id = dict()
itemSet = set()
# since all items saved in item_id format, we do not need to check the content
for outfit in allOutfits:
    set_id = outfit["set_id"]
    for item in outfit["items"]:
        item_id = item["item_id"]
        index = item["index"]
        name = "{}_{}".format(set_id, index)
        name2Id[name] = item_id
        itemSet.add(item_id)
print("Number of items: {:,}, unique items: {:,}".format(len(name2Id), len(itemSet)))


# %%
def load_negatives(fn, min_size, max_size):
    tuples = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        label, *outfit = line.split()
        size = len(outfit)
        if size < min_size or size > max_size or int(label) == 1:
            continue
        tpl = [-1] * (2 * max_size)
        valid = [item in name2Id for item in outfit]
        if not all(valid):
            continue
        for n, item in enumerate(outfit):
            item_id = name2Id[item]
            item_type = semanticDict[itemType[item_id]]
            tpl[n] = itemIndex[item_type][item_id]
            tpl[n + max_size] = item_type
        tuples.append([0, size] + tpl)
    return np.array(tuples)


# %%
train_neg = load_negatives(os.path.join(dataDir, "compatibility_train.txt"), MIN_TRAIN_SIZE, MAX_TRAIN_SIZE)
print("Numbe of train negative outfits: {:,}".format(len(train_neg)))
torchutils.io.save_csv(f"{saveDir}/train_neg", train_neg)

valid_neg = load_negatives(os.path.join(dataDir, "compatibility_valid.txt"), MIN_VALID_SIZE, MAX_VALID_SIZE)
print("Numbe of valid negative outfits: {:,}".format(len(valid_neg)))
torchutils.io.save_csv(f"{saveDir}/valid_neg", valid_neg)

test_neg = load_negatives(os.path.join(dataDir, "compatibility_test.txt"), MIN_TEST_SIZE, MAX_TEST_SIZE)
print("Numbe of test negative outfits: {:,}".format(len(test_neg)))
torchutils.io.save_csv(f"{saveDir}/test_neg", test_neg)
