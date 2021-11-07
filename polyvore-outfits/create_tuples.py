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
# dataSet = "nondisjoint"
dataSet = "disjoint"

dataDir = os.path.join(inputDir, dataSet)
saveDir = os.path.join(outputDir, dataSet)

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
trainOutfits = torchutils.io.load_json(os.path.join(dataDir, "train.json"))
validOutfits = torchutils.io.load_json(os.path.join(dataDir, "valid.json"))
testOutfits = torchutils.io.load_json(os.path.join(dataDir, "test.json"))

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
print("Number of items (max): {}".format(MAX_SIZE))


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
print("Number conditions: %d" % len(train_conditions))

data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %%
plt.hist(count_items(validOutfits), bins=np.arange(MAX_SIZE) + 1.5)
plt.title("Histogram of traning outfits")
valid_conditions = count_conditions(validOutfits)
print("Number conditions: %d" % len(valid_conditions))
data = [[i, j, "+"] for i, j in (types)]
df = pd.DataFrame(data)

df.pivot(index=0, columns=1, values=2)


# %%
plt.hist(count_items(testOutfits), bins=np.arange(MAX_SIZE) + 1.5)
plt.title("Histogram of traning outfits")
test_conditions = count_conditions(testOutfits)
print("Number conditions: %d" % len(test_conditions))
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
print("Number of items: {}, unique items: {}".format(len(name2Id), len(itemSet)))


# %%
def compatibility(fn):
    pos_tuples = []
    neg_tuples = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        label, *outfit = line.split()
        tpl = [-1] * (2 * MAX_SIZE)
        for n, item in enumerate(outfit):
            item_id = name2Id[item]
            item_type = semanticDict[itemType[item_id]]
            tpl[n] = itemIndex[item_type][item_id]
            tpl[n + MAX_SIZE] = item_type
        tpl = [0, len(outfit)] + tpl
        if int(label) == 1:
            pos_tuples.append(tpl)
        else:
            neg_tuples.append(tpl)
    return np.array(pos_tuples), np.array(neg_tuples)


# %%
splits = ["train", "valid", "test"]
outfits = dict(train=trainTuples, valid=validTuples, test=testTuples)
save = True
for phase in splits:
    compatibility_fn = os.path.join(dataDir, "compatibility_{}.txt".format(phase))
    pos_tuples, neg_tuples = compatibility(compatibility_fn)
    assert (pos_tuples == outfits[phase]).all()
    # save negative outfits into files
    print(f"Number of {phase} positive outfits: {len(pos_tuples):,}")
    print(f"Number of {phase} negative outfits: {len(neg_tuples):,}")
    if save:
        torchutils.io.save_csv(f"{saveDir}/{phase}_neg", neg_tuples)


# %%


"""Example of a question
{
    "question": ["222049137_1", "222049137_2", "222049137_3", "222049137_4", "222049137_5"],
    "blank_position": 6,
    "answers": ["136139735_5", "171518178_4", "191247707_5", "222049137_6"]
}
"""


def convert_fitb(phase, one_tuple=False):
    pos_data = []
    neg_data = []
    fn = os.path.join(dataDir, "fill_in_blank_{}.json".format(phase))
    with open(fn) as f:
        data = json.load(f)
    tuples = []
    for d in tqdm(data):
        position = d["blank_position"]
        question = d["question"]
        set_id = question[0].split("_")[0]
        question_ids = [name2Id[i] for i in question]
        question_types = [semanticDict[itemType[i]] for i in question_ids]
        question_items = [itemIndex[c][i] for c, i in zip(question_types, question_ids)]
        n = len(question) + 1
        m = MAX_SIZE - n
        one_question = []
        for ans in d["answers"]:
            items = question_items.copy()
            types = question_types.copy()
            ans_id = name2Id[ans]
            ans_type = semanticDict[itemType[ans_id]]
            ans_item = itemIndex[ans_type][ans_id]
            items.insert(position - 1, ans_item)
            types.insert(position - 1, ans_type)
            ques_tpl = [0, n] + items + [-1] * m + types + [-1] * m
            if ans.split("_")[0] == set_id:
                # true answer
                pos_data.append(ques_tpl)
                one_question.insert(0, ques_tpl)
            else:
                # false answer
                neg_data.append(ques_tpl)
                one_question.append(ques_tpl)
        tuples += one_question
    tuples = np.array(tuples)
    pos_data = np.array(pos_data)
    neg_data = np.array(neg_data)
    if one_tuple:
        return tuples
    return pos_data, neg_data


# %%
splits = ["train", "valid", "test"]
for phase in splits:
    pos, neg = convert_fitb(phase)
    print(f"Number of {phase} questions: {len(neg) // len(pos) + 1}")
    torchutils.io.save_csv(f"{saveDir}/{phase}_pos_fitb", pos)
    torchutils.io.save_csv(f"{saveDir}/{phase}_neg_fitb", neg)
