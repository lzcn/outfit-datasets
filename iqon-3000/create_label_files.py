# %%
import json
import os
from multiprocessing import Pool
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %% [markdown]
# ## IQON-3000 Dataset
#
#
# All json files are merged into one file: `processed/outfits.json`
#
# - Each entry is an outfit with keys:
#
# ```
# ['setId', 'setUrl', 'likeCount', 'user', 'items']
# ```
#
# - Each entry in `entry["items"]` is a dict with keys:
#
# ```
# ['imgUrl', 'price', 'category x color', 'itemId', 'itemName', 'itemUrl', 'breadcrumb', 'brands', 'categorys', 'options', 'colors', 'expressions']
# ```
#
#
# - Images are saved in format `images/{user}/{setId}/{itemId}_m.jpg`
#
# - There are 62 different categories
#
#

# %%
CATEGORY_MAP = {
    "Tシャツ": "top",  # "T-shirts",
    "アクセサリー": "accessories",  # "accessories",
    "アンダーウェア": "others",  # "underwear",
    "インテリア": "others",  # "interiors",
    "カーディガン": "top",  # "cardigans",
    "キャップ": "hat",  # "caps",
    "クラッチバッグ": "bag",  # "clutch bags",
    "コート": "coat",  # "coats",
    "コスメ": "others",  # "cosmetics",
    "サングラス": "accessories",  # "sunglasses",
    "サンダル": "shoes",  # "sandals",
    "ジャケット": "coat",  # "jackets",
    "ショートパンツ": "bottom",  # "shorts",
    "ショルダーバッグ": "bag",  # "shoulder bags",
    "スカート": "bottom",  # "skirts",
    "ステーショナリー": "others",  # "stationery",
    "ストール": "accessories",  # "scarves",
    "スニーカー": "shoes",  # "Sneakers",
    "ダウンジャケット": "coat",  # "down jackets",
    "タンクトップ": "top",  # "tank tops",
    "チュニック": "top",  # "tunic",
    "トートバッグ": "bag",  # "tote bags",
    "トップス": "top",  # "tops",
    "ニット": "top",  # "knits",
    "ニット帽": "hat",  # "knit hats",
    "ネイル": "others",  # "nails",
    "ネックレス": "accessories",  # "necklace",
    "パーカー": "top",  # "parkers",
    "バッグ": "bag",  # "bags",
    "ハット": "hat",  # "hat",
    "ハンドバッグ": "bag",  # handbags",
    "パンプス": "shoes",  # "pumps",
    "ピアス": "accessories",  # "piercings",
    "ブーツ": "shoes",  # "boots",
    "ファッション小物": "accessories",  # "fashion accessories",
    "ブラウス": "top",  # "blouses",
    "フレグランス": "others",  # "fragrances",
    "ブレスレット": "accessories",  # "bracelets",
    "ブローチ": "accessories",  # "broaches",
    "ヘアアクセサリー": "accessories",  # "hair accessories",
    "ベルト": "accessories",  # "belts",
    "ボストンバッグ": "bag",  # "Boston bags",
    "ボディケア": "others",  # "body care",
    "メガネ": "others",  # "glasses",
    "リュック": "bag",  # "backpacks",
    "リング": "accessories",  # "rings",
    "ルームウェア": "others",  # "room wear",
    "ルームシューズ": "others",  # "room shoes",
    "レッグウェア": "others",  # "legwear",
    "ロングスカート": "dress",  # "long skirts",
    "ロングパンツ": "bottom",  # "long pants",
    "ワンピース": "dress",  # "dresses",
    "傘": "others",  # "umbrellas",
    "小物": "others",  # "accessories",
    "帽子": "hat",  # "hats",
    "手袋": "accessories",  # "gloves",
    "水着": "others",  # "swimwear",
    "浴衣": "others",  # "yukata",
    "腕時計": "others",  # "watches",
    "財布": "others",  # "wallets",
    "靴": "shoes",  # "shoes",
}

allCates = set(CATEGORY_MAP.values())
allCates.remove("others")
# sort all categories
allCates = sorted(list(allCates))
cateMap = dict()
for cate in allCates:
    cateMap[cate] = len(cateMap)
print(cateMap)

# %%
image_dir = "raw"
label_dir = "processed"
os.makedirs(label_dir, exist_ok=True)

# %%
# open raw json dataset
json_file = "processed/outfits.json"
with open(json_file, "r") as f:
    allOutfits = json.load(f)

# %%
print("Number of outfits:", len(allOutfits))
# Example outfit
print("Outfit keys:")
pprint(list(allOutfits[0].keys()))
print("Item keys:")
pprint(list(allOutfits[0]["items"][0].keys()))

# %% [markdown]
# ## Step-1: Clean outfit
#
# outfit that has 3 ~ 8 items will be retained

# %%
# plot histogram of outfit size
item_size = [len(outfit["items"]) for outfit in allOutfits]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(item_size, np.array(range(min(item_size) - 1, max(item_size) + 1, 1)) + 0.5, density=True)
ax.set_title("Historgram of outfit size")
ax.set_xlabel("Number of items")
ax.set_ylabel("Number of outfits")

# %%
# clean outfit
min_size, max_size = 3, 8


def clean_outfits(outfits: list, min_size=3, max_size=8):
    cleaned_outfits = []
    # each outfit is a dict
    for outfit in tqdm(outfits):
        cleaned_items = []
        # each item is a dict
        cate_set = set()
        for item in outfit["items"]:
            # get the category name in Japanese
            cate = item["category x color"].split("\u00d7")[0].strip()
            # get the category name in English
            if CATEGORY_MAP.get(cate, "others") in allCates:
                # if it in pre-defined set, then add the item to list
                item["cate"] = CATEGORY_MAP[cate]
                cate_set.add(item["cate"])
                cleaned_items.append(item)
        # if cleaned outfit has valid size
        if min_size <= len(cleaned_items) <= max_size and len(cate_set) > 2:
            outfit["items"] = cleaned_items
            cleaned_outfits.append(outfit)
    print("Number of outfits before clean: {}".format(len(outfits)))
    print("Number of outfits after clean: {}".format(len(cleaned_outfits)))
    return cleaned_outfits


cleanedOutfits = clean_outfits(allOutfits, min_size, max_size)

# %%
# plot the histogram of cleaned version
item_size = [len(outfit["items"]) for outfit in cleanedOutfits]
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(item_size, np.array(range(min(item_size) - 1, max(item_size) + 1, 1)) + 0.5, density=True)
ax.set_title("Historgram of outfit size (cleaned)")
ax.set_xlabel("Number of items")
ax.set_ylabel("Number of outfits")

# %% [markdown]
# ## Step-2: Convert outfit list to dict

# %%
# convert outfit list to user set


def convert(outfits):
    user_set = dict()
    for outfit in tqdm(outfits):
        set_id = int(outfit["setId"])
        user_id = int(outfit["user"])
        if user_id not in user_set:
            user_set[user_id] = []
        items = []
        for item in outfit["items"]:
            item_id = int(item["itemId"])
            item_type = cateMap[item["cate"]]
            items.append(
                dict(
                    item_id=item_id,
                    item_type=item_type,
                )
            )
        user_set[user_id].append(
            dict(
                set_id=set_id,
                items=items,
            )
        )
    return user_set


# %%
userOutfitSet = convert(cleanedOutfits)
print("Number of users: {}".format(len(userOutfitSet)))

# %%
# show one example
one_outfit = list(userOutfitSet.values())[0][0]
user_id = list(userOutfitSet.keys())[0]
print("user_id: ", user_id)
pprint(one_outfit)

# %%
# count number of categories
def count_category(dataset):
    cate_set = set()
    for _, sets in dataset.items():
        for outfit in sets:
            for item in outfit["items"]:
                cate_set.add(item["item_type"])
    return cate_set


# %%
cate_set = count_category(userOutfitSet)
print("Number of category: {}".format(len(cate_set)))

# %%
def plot_num_outfits(dataset):
    num_oufits = np.array([len(v) for v in dataset.values()])
    min_size = min(num_oufits)
    max_size = max(num_oufits)
    print("Number of outfits min: {}, max: {}".format(min_size, max_size))
    ranges = np.array(range(min_size - 1, max_size + 1, 10)) + 0.5

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(num_oufits, ranges)
    ax.set_xlabel("Number of outfits")
    ax.set_ylabel("Number of users")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(num_oufits, ranges, cumulative=True, density=True)
    ax.set_xlabel("Number of outfits")
    ax.set_ylabel("Percentage of users")


# %%
plot_num_outfits(userOutfitSet)

# %% [markdown]
# ## Step-3: Clean users
#
# user that has more than 120 outfit will be retained.

# %%
# 1. clean users that has less than 120 outfit
def clean_outfits(dataset):
    cleaned_user_set = dict()
    num_outfits = 120
    cnt = 0
    for user, sets in tqdm(dataset.items()):
        if len(sets) >= num_outfits:
            cleaned_user_set[user] = sets
    print("Number of users before: {:,}".format(len(dataset)))
    print("Number of users after: {:,}".format(len(cleaned_user_set)))
    return cleaned_user_set


cleanedUserSet = clean_outfits(userOutfitSet)
cate_set = count_category(cleanedUserSet)
print("Number of categories: {}".format(len(cate_set)))

# %%
plot_num_outfits(cleanedUserSet)

# %%
def split_outfits(dataset, train=85, val=15, test=20):
    train_user_set, val_user_set, test_user_set = dict(), dict(), dict()
    for user, sets in tqdm(dataset.items()):
        sets = np.random.permutation(sets).tolist()
        train_user_set[user] = sets[:train]
        val_user_set[user] = sets[train : val + train]
        test_user_set[user] = sets[val + train : train + val + test]
    return dict(train=train_user_set, val=val_user_set, test=test_user_set)


splitUserSet = split_outfits(cleanedUserSet)
phases = list(splitUserSet.keys())

# %%
for k, v in splitUserSet.items():
    print("Number of users in {}: {}".format(k, len(v)))
    print("Number of categories in {}: {}".format(k, len(count_category(v))))

# %%
# leave outf users for cold-start problem
def split_user(dataset, user_idxs):
    x, y = dict(), dict()
    for user, sets in dataset.items():
        if user in user_idxs:
            x[user] = sets
        else:
            y[user] = sets
    return x, y


# %%
num_all_users = len(cleanedUserSet)
num_old_users = 550
num_new_users = num_all_users - num_old_users

user_idxs = np.random.permutation(list(cleanedUserSet.keys())).tolist()[:num_old_users]
# for regular task
oldUserSplit = dict()
# for cold-start task
newUserSplit = dict()

for phase, outfit_split in splitUserSet.items():
    x, y = split_user(outfit_split, user_idxs)
    oldUserSplit[phase] = x
    newUserSplit[phase] = y
    # no user overlap
    assert len(set(x.keys()) & set(y.keys())) == 0
    print("Number of categories in old user train: {}".format(len(count_category(x))))
    print("Number of categories in new user train: {}".format(len(count_category(y))))

# %%
# plot number of items of each category
def count_cate_set(dataset):
    cate_cnt = [0] * len(cateMap)
    for user, outfits in dataset.items():
        for oft in outfits:
            for item in oft["items"]:
                cate_cnt[item["item_type"]] += 1
    return cate_cnt


cate_count = []
for dataset in oldUserSplit.values():
    cate_count.append(count_cate_set(dataset))
for dataset in newUserSplit.values():
    cate_count.append(count_cate_set(dataset))
plt.close()
figure = plt.figure()
ax = figure.add_subplot(111)
for cnt in cate_count:
    ax.plot(np.array(cnt) / sum(cnt))

# %%
def get_user_idx_map(split_data):
    user_idx_set = set()
    for dataset in split_data.values():
        user_idx_set |= set(dataset.keys())
    user_idx_map = {user_id: i for i, user_id in enumerate(user_idx_set)}
    return user_idx_map


def convert_tuples(dataset, user_idx_map):
    max_item = 8
    tuples = []
    for user, outfits in dataset.items():
        idx = user_idx_map[user]
        for outfit in outfits:
            tpl = [-1] * 8
            cate = [-1] * 8
            for n, item in enumerate(outfit["items"]):
                tpl[n] = item["item_id"]
                cate[n] = item["item_type"]
            tuples.append([idx] + tpl + cate)
    return tuples


# %%
def delete_split_overlap(split, num_users=550):
    def to_array(split_set):
        array = np.array(list(split_set))
        array = array[array[:, 0].argsort()]
        return array

    train_set = set(map(tuple, split["train"]))
    valid_set = set(map(tuple, split["val"]))
    test_set = set(map(tuple, split["test"]))
    print("Before")
    print("Number of train: {:.3f}".format(len(train_set) / num_users))
    print("Number of valid: {:.3f}".format(len(valid_set) / num_users))
    print("Number of test: {:.3f}".format(len(test_set) / num_users))
    # deleta overlap outfits
    test_set = test_set - train_set - valid_set
    valid_set = valid_set - test_set - train_set
    print("After")
    print("Number of train: {:.3f}".format(len(train_set) / num_users))
    print("Number of valid: {:.3f}".format(len(valid_set) / num_users))
    print("Number of test: {:.3f}".format(len(test_set) / num_users))
    return dict(train=to_array(train_set), val=to_array(valid_set), test=to_array(test_set))


# %%
user_idx_map = get_user_idx_map(oldUserSplit)
oldUserTuples = dict()
for key, dataset in oldUserSplit.items():
    oldUserTuples[key] = convert_tuples(dataset, user_idx_map)
oldUserTuples = delete_split_overlap(oldUserTuples, num_old_users)

# %%
user_idx_map = get_user_idx_map(newUserSplit)
newUserTuples = dict()
for key, dataset in newUserSplit.items():
    newUserTuples[key] = convert_tuples(dataset, user_idx_map)
newUserTuples = delete_split_overlap(newUserTuples, num_new_users)

# %%
def load_tuples(file):
    return np.array(pd.read_csv(file, dtype=np.int, header=None))


# load splits from previous file
oldUserTuples = delete_split_overlap(
    {
        "train": load_tuples("./label/train_550_pos.txt"),
        "val": load_tuples("label/val_550_pos.txt"),
        "test": load_tuples("label/test_550_pos.txt"),
    },
    550,
)
newUserTuples = delete_split_overlap(
    {
        "train": load_tuples("./label/train_58_pos.txt"),
        "val": load_tuples("label/val_58_pos.txt"),
        "test": load_tuples("label/test_58_pos.txt"),
    },
    58,
)

# %%
import csv

if True:
    for phase, tuples in oldUserTuples.items():
        with open("processed/{}_{}_pos.txt".format(phase, num_old_users), "w") as f:
            writer = csv.writer(f)
            writer.writerows(tuples)

    for phase, tuples in newUserTuples.items():
        with open("processed/{}_{}_pos.txt".format(phase, num_new_users), "w") as f:
            writer = csv.writer(f)
            writer.writerows(tuples)

# %%
def get_item_list(pos_tuples):
    item_set = [set() for _ in range(8 + 1)]
    item_ids, item_types = np.split(pos_tuples[:, 1:], 2, axis=1)
    for idxs, types in zip(item_ids, item_types):
        for idx, c in zip(idxs, types):
            item_set[c].add(idx)
    return [np.array(list(s)) for s in item_set]


def generate_neg_tuples(pos_tuples, ratio=10):
    neg_tpls = []
    item_set = get_item_list(pos_tuples)
    for tpl in tqdm(pos_tuples):
        for _ in range(ratio):
            neg_tpl = [-1] * 17
            user_id = tpl[0]
            neg_tpl[0] = user_id
            items = tpl[1:9]
            cate = tpl[9:]
            neg_tpl[9:] = tpl[9:]
            for i, cate in enumerate(tpl[9:]):
                if cate == -1:
                    break
                idx = np.random.choice(item_set[cate])
                neg_tpl[i + 1] = idx
            neg_tpls.append(neg_tpl)
    return np.array(neg_tpls)


# %%
for p in ["train", "val", "test"]:
    pos_tpls = oldUserTuples[p]
    neg_tpls = generate_neg_tuples(pos_tpls)
    pos_item_list = get_item_list(pos_tpls)
    neg_item_list = get_item_list(neg_tpls)
    print([set(neg_item_list[i]) - set(pos_item_list[i]) for i in range(9)])
    with open("processed/{}_{}_neg.txt".format(p, num_old_users), "w") as f:
        writer = csv.writer(f)
        writer.writerows(neg_tpls)

# %%
for p in ["train", "val", "test"]:
    pos_tpls = newUserTuples[p]
    neg_tpls = generate_neg_tuples(pos_tpls)
    pos_item_list = get_item_list(pos_tpls)
    neg_item_list = get_item_list(neg_tpls)
    print([set(neg_item_list[i]) - set(pos_item_list[i]) for i in range(9)])
    with open("processed/{}_{}_neg.txt".format(p, num_new_users), "w") as f:
        writer = csv.writer(f)
        writer.writerows(neg_tpls)

# %%
