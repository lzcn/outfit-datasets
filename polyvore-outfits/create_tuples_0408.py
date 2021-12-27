# %%
import csv
import json
import os
import pickle as pkl
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchutils
from tqdm import tqdm

# %%
inputDir = "release"
outputDir = "processed"
dataSet = "nondisjoint"
# dataSet = "disjoint"
minSize = 4
maxSize = 8
dataDir = os.path.join(inputDir, dataSet)
saveDir = os.path.join(outputDir, dataSet + "0408")

os.makedirs(saveDir, exist_ok=True)

saveFile = True
# %%
# all imgages saved in item_id.jpg format
image_files = torchutils.files.scan_files(f"{inputDir}/images", suffix="jpg")
print("Nubmer of images: {:,}".format(len(image_files)))

# %% [markdown]
# ## Example item format
# ```json
# {'category_id': '29',
#  'catgeories': ["Women's Fashion", 'Clothing', 'Shorts'],
#  'description': 'Blue High Waist Plain Polyester Loose Button Fly Street '
#                 'Shorts, Size Available: S,M,L,XL S, M, L, XL Plain Blue '
#                 'Polyester Style: Street.',
#  'related': ['High waisted jean shorts',
#              'High-waisted denim shorts',
#              'Blue shorts',
#              'High-rise shorts',
#              'Highwaist shorts',
#              'Button shorts'],
#  'semantic_category': 'bottoms',
#  'title': 'Blue High Waist Buttons Denim Shorts',
#  'url_name': 'blue high waist buttons denim'}
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
pprint(semanticDict)

# %% [markdown]
# # Reformat the outfit tuples
#

# %%


with open(f"{inputDir}/{dataSet}/typespaces.p", "rb") as f:
    types = pkl.load(f)

print("Number of conditions:", len(types))
data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %%


def clean_outfits(outfits, min_size, max_size):
    results = []
    for outfit in outfits:
        items = outfit["items"]
        size = len(items)
        if min_size <= size <= max_size:
            results.append(outfit)
    return results


trainOutfits = torchutils.io.load_json(os.path.join(dataDir, "train.json"))
validOutfits = torchutils.io.load_json(os.path.join(dataDir, "valid.json"))
testOutfits = torchutils.io.load_json(os.path.join(dataDir, "test.json"))

trainOutfits = clean_outfits(trainOutfits, minSize, maxSize)
validOutfits = clean_outfits(validOutfits, minSize, maxSize)
testOutfits = clean_outfits(testOutfits, minSize, maxSize)

allOutfits = trainOutfits + validOutfits + testOutfits
outfit = np.random.choice(allOutfits)
print(
    "Number of {} outfits: {:,} = train ({:,}) + valid({:,}) + test({:,})".format(
        dataSet, len(allOutfits), len(trainOutfits), len(validOutfits), len(testOutfits)
    )
)
print("Example outfit")
pprint(outfit)

MAX_SIZE = 0
for outfit in allOutfits:
    MAX_SIZE = max(MAX_SIZE, len(outfit["items"]))
print("Max number of items in outfit: {}".format(MAX_SIZE))


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

# sort items to get consistent order
itemList = [sorted(list(items)) for items in itemSet]
itemIndex = [{key: index for index, key in enumerate(items)} for items in itemList]
print("Number of items in each category: ", list(map(len, itemList)))

if saveFile:
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
plt.hist(count_items(trainOutfits), bins=np.arange(MAX_SIZE) + 1.5)
plt.title("Histogram of traning outfits")
train_conditions = count_conditions(trainOutfits)
print("Number conditions (train): %d" % len(train_conditions))

data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %%
plt.hist(count_items(validOutfits), bins=np.arange(MAX_SIZE) + 1.5)
plt.title("Histogram of traning outfits")
valid_conditions = count_conditions(validOutfits)
print("Number conditions (valid): %d" % len(valid_conditions))
data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %%
plt.hist(count_items(testOutfits), bins=np.arange(MAX_SIZE) + 1.5)
plt.title("Histogram of traning outfits")
test_conditions = count_conditions(testOutfits)
print("Number conditions (test): %d" % len(test_conditions))
data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)
df.pivot(index=0, columns=1, values=2)


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
trainTuples = convert_tuples(trainOutfits, MAX_SIZE)
validTuples = convert_tuples(validOutfits, MAX_SIZE)
testTuples = convert_tuples(testOutfits, MAX_SIZE)


# %%
with open(os.path.join(saveDir, "train_pos"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(trainTuples)

with open(os.path.join(saveDir, "valid_pos"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(validTuples)

with open(os.path.join(saveDir, "test_pos"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(testTuples)
