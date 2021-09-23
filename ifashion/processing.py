#!/usr/bin/env python
# %%
import os
import torchutils
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# %%

dataDir = "raw"
outputDir = "processed"
os.makedirs(outputDir, exist_ok=True)

# %%

# convert files
item_data_fn = f"{outputDir}/item_data.json"
item_cate_fn = f"{outputDir}/item_cate.json"
item_list_fn = f"{outputDir}/item_list.json"
cate_list_fn = f"{outputDir}/cate_list.json"
if os.path.exists(item_data_fn) and os.path.exists(item_cate_fn) and os.path.exists(item_list_fn):
    print("load files from cache")
    itemData = torchutils.io.load_json(item_data_fn)
    itemCate = torchutils.io.load_json(item_cate_fn)
    itemList = torchutils.io.load_json(item_list_fn)
    cateList = torchutils.io.load_json(cate_list_fn)
else:
    itemCate = dict()
    itemData = dict()
    itemList = dict()
    with open(f"{dataDir}/item_data.txt") as f:
        for line in f:
            item_id, cate, *title = line.strip().split(",")
            itemData[item_id] = {"cate": cate, "title": " ".join(title)}
            if cate not in itemCate:
                itemCate[cate] = set()
            itemCate[cate].add(item_id)
    for cate, items in itemCate.items():
        itemCate[cate] = list(items)
    itemList = sorted(list(itemData.keys()))
    cateList = sorted(list(itemCate.keys()))
    torchutils.io.save_json(item_cate_fn, itemCate)
    torchutils.io.save_json(item_data_fn, itemData)
    torchutils.io.save_json(item_list_fn, itemList)
    torchutils.io.save_json(cate_list_fn, cateList)

num_items = np.sort([len(items) for items in itemCate.values()])[::-1]
fig = plt.figure()
plt.plot(num_items)
plt.savefig(f"{outputDir}/category_size.pdf")

print("{:=^25}".format("Item Data"))
print("Number of items: {:,}".format(len(itemData)))
print("Number of categories: {:,}".format(len(itemCate)))

# ========Item Data========
# Number of items: 4,747,039
# Number of categories: 75

# %%
outfit_data_fn = f"{outputDir}/outfit_data.json"
outfit_list_fn = f"{outputDir}/outfit_list.json"
if os.path.exists(outfit_data_fn) and os.path.exists(outfit_list_fn):
    outfitData = torchutils.io.load_json(outfit_data_fn)
    outfitList = torchutils.io.load_json(outfit_list_fn)
else:
    outfitData = dict()
    with open("raw/outfit_data.txt") as f:
        for line in f:
            outfit_id, items = line.strip().split(",")
            outfitData[outfit_id] = items.split(";")
    outfitList = sorted(outfitData.keys())
    torchutils.io.save_json(outfit_data_fn, outfitData)
    torchutils.io.save_json(outfit_list_fn, outfitList)

num_items = []
item_set = list()
for outfit_id, items in outfitData.items():
    item_set += items
    num_items.append(len(items))
item_set = set(item_set)

fig = plt.figure()
plt.hist(num_items, density=True)
plt.savefig(f"{outputDir}/histogram_outfit_length.pdf")

print("{:=^25}".format("Outfit Data"))
print("Number of outfits: {:,}".format(len(outfitData)))
print("Number of items: {:,}. [{} - {}]".format(sum(num_items), min(num_items), max(num_items)))
print("Number of unique items: {:,}".format(len(item_set)))
print("Ratio of reuse: {:.3f}".format(sum(num_items) / len(item_set)))
print("Number of items/outfit: {:.3f}".format(sum(num_items) / len(outfitData)))

# =======Outfit Data=======
# Number of outfits: 1,013,136
# Number of items: 3,692,882. [3 - 9]
# Number of unique items: 583,464
# Ratio of reuse: 6.329
# Number of items/outfit: 3.645

# %%
user_click_fn = f"{outputDir}/user_click.json"
user_list_fn = f"{outputDir}/user_list.json"
user_outfit_fn = f"{outputDir}/user_outfit.json"
if os.path.exists(user_outfit_fn) and os.path.exists(user_list_fn):
    userClick = torchutils.io.load_json(user_click_fn)
    userList = torchutils.io.load_json(user_list_fn)
    userOutfit = torchutils.io.load_json(user_outfit_fn)
else:
    userClick = dict()
    userOutfit = dict()
    userList = []
    with open(f"{dataDir}/user_data.txt") as f:
        for line in f:
            user, clicks, outfit = line.strip().split(",")
            clicks = clicks.split(";")
            if user not in userClick:
                userClick[user] = []
            if user not in userOutfit:
                userOutfit[user] = []
            userClick[user].append(clicks)
            userOutfit[user].append(outfit)
    userList = sorted(userOutfit.keys())
    torchutils.io.save_json(user_click_fn, userClick)
    torchutils.io.save_json(user_outfit_fn, userOutfit)
    torchutils.io.save_json(user_list_fn, userList)

clicks = []
outfits = []
num_outfits = []
num_clicks = []
for user_id in tqdm(userList, desc="Counting users"):
    o = userOutfit[user_id]
    c = userClick[user_id]
    num_outfits.append(len(o))
    outfits += o
    clicks += [i for n in c for i in n]
    num_clicks += [len(n) for n in c]
clicks = set(clicks)
outfits = set(outfits)


fig = plt.figure()
plt.plot(np.sort(num_outfits)[::-1])
plt.savefig(f"{outputDir}/number_of_outfits.pdf")

print("{:=^25}".format("User Data"))
print("Number of users: {:,}".format(len(userList)))
print("Number of unique outfits: {:,}".format(len(outfits)))
print("Number of outfits: {:,}".format(sum(num_outfits)))
print("Ratio of reuse outfits: {:,}".format(sum(num_outfits) / len(outfits)))
print("Number of unique clicked items: {:,}".format(len(clicks)))
print("Number of clicks: {:,}".format(sum(num_clicks)))
print("Ratio of reuse clicked items: {:.3f}".format(sum(num_clicks) / len(clicks)))
print("Number of clicks/outfit: {:.3f}".format(sum(num_clicks) / sum(num_outfits)))

# ========User Data========
# Number of users: 3,569,112
# Number of unique outfits: 127,169
# Number of outfits: 19,191,117
# Ratio of reuse outfits: 150.910
# Number of unique clicked items: 4,463,302
# Number of clicks: 620,880,101
# Ratio of reuse clicked items: 139.108
# Number of clicks/outfit: 32.352

# %%

# convert user to tuples
item_index = {v: i for i, v in enumerate(itemList)}
outfit_index = {v: i for i, v in enumerate(outfitList)}
user_index = {v: i for i, v in enumerate(userList)}
cate_index = {v: i for i, v in enumerate(cateList)}


max_size = 9
tuple_user_item = []
tuple_user_outfit = []
tuple_user_click = []
for user in tqdm(userList, desc="Converting"):
    uidx = user_index[user]
    outfits = userOutfit[user]
    for outfit_id in outfits:
        outfit = outfitData[outfit_id]
        size = len(outfit)
        append = [-1] * (max_size - size)
        item = [item_index[i] for i in outfit] + append
        cate = [cate_index[itemData[i]["cate"]] for i in outfit] + append
        tuple_user_item.append([[uidx, size] + item + cate])
        tuple_user_outfit.append([uidx, outfit_index[outfit_id]])
    clicks = userClick[user]
    for seq in clicks:
        tuple_user_click.append([uidx] + [item_index[i] for i in seq])

torchutils.io.save_csv(f"{outputDir}/tuple_user_item.csv", tuple_user_item)
torchutils.io.save_csv(f"{outputDir}/tuple_user_outfit.csv", tuple_user_outfit)
torchutils.io.save_csv(f"{outputDir}/tuple_user_click.csv", tuple_user_click)
