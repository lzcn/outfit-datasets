# %%
import csv
import json
import os
import pprint

import numpy as np
import PIL
import torchutils
from tqdm.auto import tqdm

# %% [markdown]
# ## Outfit dataset
# Outfits are saved in three splits:
#
# ```json
# train: "train_no_dup.json",
# valid: "valid_no_dup.json",
# test: "test_no_dup.json",
# ```
#
# Each outfit in the `json` file has the following keys:

# ```json
# ['name', 'views', 'items', 'image', 'likes', 'date', 'set_url', 'set_id', 'desc']
# ```

# Each item in an outfit has the following keys:

# ```json
# ['index', 'name', 'prices', 'likes', 'image', 'categoryid']
# ```


# %%


def get_item_type(item):
    return item["index"] - 1


def get_item_id(item):
    return item["image"].split("tid=")[-1]


def load_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


# %%

inputDir = "maryland-polyvore/release"
outputDir = "maryland-polyvore/processed"

CHECK_IMAGES = False
MAKE_LMDB = False

# %% [markdown]
# ## Load outfits

# %%
trainOutfits = load_json(f"{inputDir}/label/train_no_dup.json")
validOutfits = load_json(f"{inputDir}/label/valid_no_dup.json")
testOutfits = load_json(f"{inputDir}/label/test_no_dup.json")

allOutfits = trainOutfits + validOutfits + testOutfits
print(
    "Number of outfits: {} = {} + {} + {}".format(
        len(allOutfits), len(trainOutfits), len(validOutfits), len(testOutfits)
    )
)


# %%
print("Example outfit:")
pprint.pprint(allOutfits[0], depth=1)
print("Example item:")
pprint.pprint(allOutfits[0]["items"][0])


# %%
MAX_SIZE = 8

itemSet = set()
itemImages = dict()
categorySet = set()
outfitSet = set()
# {set_id}_{index}: {item_id}
name2Id = dict()

# item index in item list
itemReIndex = []
itemList = [set() for _ in range(MAX_SIZE)]

num_items = 0
for outfit in allOutfits:
    set_id = outfit["set_id"]
    outfitSet.add(set_id)
    num_items += len(outfit["items"])
    for item in outfit["items"]:
        categorySet.add(item["categoryid"])
        index = item["index"]
        name = "{}_{}".format(set_id, index)
        # the unique id of item
        item_id = get_item_id(item)
        item_type = get_item_type(item)
        name2Id[name] = item_id
        itemSet.add(item_id)
        # use index as item category
        itemList[item_type].add(item_id)
        if item_id not in itemImages:
            itemImages[item_id] = []
        itemImages[item_id].append(os.path.join(f"{inputDir}/images", "{}/{}.jpg".format(set_id, index)))

print("Number of unique items: {:,}".format(len(itemSet)))
print("Reuse ratio: {:.3f} = {:,} / {:,}".format(num_items / len(itemSet), num_items, len(itemSet)))
print("Number of categories: {}".format(len(categorySet)))
print("Average number of items in an outfit: {:.2f}".format(num_items / len(allOutfits)))

# %% [markdown]
# ## Create item list

# %%
# convert to item list
itemReIndex = []
itemList = [list(items) for items in itemList]
for i in range(MAX_SIZE):
    items = itemList[i]
    item_index = {item_id: i for i, item_id in enumerate(items)}
    itemReIndex.append(item_index)

torchutils.io.save_json(f"{outputDir}/original/items.json", itemList)

# %% [markdown]
# ## Check items
#
# Since one image maybe used multiple times. Check whether the content is the same.

# %%
# all compatibility outfits are in allOutfits
# name format: {set_id}_{item_index}

with open(f"{inputDir}/label/fashion_compatibility_prediction.txt") as f:
    lines = f.readlines()
for line in lines:
    set_id = line.split(" ")[1].split("_")[0]
    assert set_id in outfitSet

# all fitb outfits are in allOutfits
# name format: {set_id}_{item_index}
# {"question": [names, ...], "answers": [names, ], "blank_position": n}
with open(f"{inputDir}/label/fill_in_blank_test.json") as f:
    data = json.load(f)
for d in data:
    position = d["blank_position"]
    question = d["question"]
    for q in question:
        set_id = q.split("_")[0]
        assert set_id in outfitSet


# %%
def check_reuse_images():
    itemImages = dict()
    for k, v in tqdm(itemImages.items()):
        if len(v) > 1:
            images = []
            for fn in v:
                with open(fn, "rb") as f:
                    images.append(np.array(PIL.Image.open(f).convert("RGB")))
            itemImages[k] = images

    error = dict()
    for k, v in tqdm(itemImages.items()):
        imgs = np.stack(v)
        mean = (imgs - imgs.mean(axis=0)).mean()
        error[k] = mean
    print("Mean error: {:.3f}".format(np.array(list(error.values())).mean()))


if CHECK_IMAGES:
    check_reuse_images()

# %% [markdown]
# ## Convert to outfit to tuples

# %%


def convert_to_tuples(outfits):
    tuples = []
    for outfit in outfits:
        items = [-1] * MAX_SIZE
        types = [-1] * MAX_SIZE
        size = len(outfit["items"])
        for i, item in enumerate(outfit["items"]):
            tid = item["image"].split("tid=")[-1]
            item_type = item["index"] - 1
            item_index = itemReIndex[item_type][tid]
            items[i] = item_index
            types[i] = item_type
        tuples.append([0] + [size] + items + types)
    return np.array(tuples)


# %%
train_pos = convert_to_tuples(trainOutfits)
valid_pos = convert_to_tuples(validOutfits)
test_pos = convert_to_tuples(testOutfits)


# %%

# save all positive outfits
torchutils.io.save_csv(f"{outputDir}/original/train_pos", train_pos)
torchutils.io.save_csv(f"{outputDir}/original/valid_pos", valid_pos)
torchutils.io.save_csv(f"{outputDir}/original/test_pos", test_pos)

# %% [markdown]
# ## Convert FITB and Negative tuples

# %%


def convert_compatibility(data):
    eval_pos = []
    eval_neg = []

    for line in data:
        label, *names = line.split()
        size = len(names)
        m = MAX_SIZE - size
        types = [int(name.split("_")[1]) - 1 for name in names]
        items = [itemReIndex[c][name2Id[i]] for i, c in zip(names, types)]
        tpl = [0, size] + items + [-1] * m + types + [-1] * m
        if int(label) == 1:
            eval_pos.append(tpl)
        else:
            eval_neg.append(tpl)
    eval_pos = np.array(eval_pos)
    eval_neg = np.array(eval_neg)
    return eval_pos, eval_neg


# %%
with open(f"{inputDir}/label/fashion_compatibility_prediction.txt") as f:
    lines = f.readlines()
eval_pos, eval_neg = convert_compatibility(lines)
# the positive tuples should match those in compatibility
assert (test_pos == eval_pos).all()
print("Number of positive outfits: {:,}".format(len(eval_pos)))
print("Number of negative outfits: {:,}".format(len(eval_neg)))


# %%
torchutils.io.save_csv(f"{outputDir}/original/test_neg", eval_neg)

# %%


def convert_fitb(data):
    tuples = []
    for d in data:
        position = d["blank_position"]
        question = d["question"]
        question_types = [int(s.split("_")[1]) - 1 for s in question]
        question_items = [itemReIndex[c][name2Id[i]] for i, c in zip(question, question_types)]
        size = len(question) + 1
        m = MAX_SIZE - size
        for answer in d["answers"]:
            c = int(answer.split("_")[-1]) - 1
            i = itemReIndex[c][name2Id[answer]]
            items = question_items.copy()
            types = question_types.copy()
            items.insert(position - 1, i)
            types.insert(position - 1, c)
            tuples.append([0, size] + items + [-1] * m + types + [-1] * m)
    tuples = np.array(tuples)
    return tuples


# %%
with open(f"{inputDir}/label/fill_in_blank_test.json") as f:
    data = json.load(f)

test_fitb = convert_fitb(data)
torchutils.io.save_csv(f"{outputDir}/original/test_fitb", test_fitb)

# %% [markdown]
# ## Tuples from type-aware embedding

# %%

dst_dir = f"{outputDir}/hardneg"
src_dir = f"{inputDir}/maryland_polyvore_hardneg/"

torchutils.io.save_json(f"{dst_dir}/items.json", itemList)


# %%
splits = ["train", "valid", "test"]
outfits = dict(train=train_pos, valid=valid_pos, test=test_pos)
for phase in splits:
    fn = os.path.join(src_dir, "compatibility_{}.txt".format(phase))
    with open(fn) as f:
        data = f.readlines()
    pos_tuples, neg_tuples = convert_compatibility(data)
    assert (pos_tuples == outfits[phase]).all()
    torchutils.io.save_csv(os.path.join(dst_dir, "{}_neg".format(phase)), neg_tuples)
    torchutils.io.save_csv(os.path.join(dst_dir, "{}_pos".format(phase)), pos_tuples)
    print("Number of positive outfits ({}): {:,}".format(phase, len(pos_tuples)))
    print("Number of negative outfits ({}): {:,}".format(phase, len(neg_tuples)))


# %%
splits = ["train", "valid", "test"]
for phase in splits:
    data = torchutils.io.load_json(os.path.join(src_dir, "fill_in_blank_{}.json".format(phase)))
    test_fitb = convert_fitb(data)
    torchutils.io.save_csv(os.path.join(dst_dir, "{}_fitb".format(phase)), test_fitb)
    print("Number of questions ({}): {:,}".format(phase, len(test_fitb) // 4))
