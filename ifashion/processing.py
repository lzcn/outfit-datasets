#!/usr/bin/env python
# %%
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_json(file: Union[str, Path]) -> Any:
    """Load JSON file.

    Args:
        file: JSON file path

    Returns:
        Parsed JSON object

    Example:
        >>> config = load_json("config.json")
        >>> print(config["name"])
    """
    file = Path(file).expanduser()
    with file.open("r") as f:
        return json.load(f)


def save_json(file: Union[str, Path], data: Any, overwrite: bool = False) -> None:
    """Save data to a JSON file.

    Args:
        file: File path
        data: Data to be serialized
        overwrite: If False and file exists, will skip saving

    Example:
        >>> data = {"name": "model", "version": 1}
        >>> save_json("output.json", data, overwrite=True)
    """
    file = Path(file).expanduser()
    if file.exists() and not overwrite:
        logger.warning("%s already exists. Skipped.", file)
        return
    with file.open("w") as f:
        json.dump(data, f)


def load_csv(
    file: Union[str, Path],
    skip_rows: int = 0,
    converter: Optional[Callable[[str], Any]] = None,
) -> List[List[Any]]:
    """Load CSV file.

    Args:
        file: File path
        skip_rows: Rows to skip from top
        converter: Optional callable to convert each element

    Returns:
        Parsed list of rows

    Example:
        >>> rows = load_csv("data.csv")
        >>> rows[0]
        ['id', 'score']

        >>> rows = load_csv("data.csv", converter=int)
        >>> rows[1]
        [1, 95]
    """
    file = Path(file).expanduser()
    with file.open("r", newline="") as f:
        reader = csv.reader(f)
        for _ in range(skip_rows):
            next(reader)
        data = list(reader)
        if converter:
            data = [list(map(converter, row)) for row in data]
    return data


def save_csv(file, data, header=None, overwrite=False) -> None:
    """Save data to CSV file.

    Args:
        file: File path
        data: List of rows
        header: Optional list of column names
        overwrite: If False and file exists, will skip saving

    Example:
        >>> rows = [[1, 90], [2, 85]]
        >>> save_csv("scores.csv", rows, header=["id", "score"], overwrite=True)
    """
    file = Path(file).expanduser()
    if file.exists() and not overwrite:
        logger.warning("%s already exists. Skipped.", file)
        return

    with file.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


# %% Setup the directories

dataDir = "release"
imageDir = "images"
outputDir = "processed"
os.makedirs(outputDir, exist_ok=True)
os.makedirs(os.path.join(outputDir, "josn"), exist_ok=True)

# %% Process the `item_data.txt` file
""""
The item_data.txt saves each item in the following format:

    item_id,cate_id,pic_url,title

We will convert this file into the following:

- itemData: {item_id: {"cate": cate_id, "url": pic_url, "title": title}}
- itemList: [item_id]
- cateItem: {cate_id: [item_id]}
- cateList: [cate_id]


"""


# convert files
item_data_fn = f"{outputDir}/json/item_data.json"
item_list_fn = f"{outputDir}/json/item_list.json"
cate_item_fn = f"{outputDir}/json/cate_item.json"
cate_list_fn = f"{outputDir}/json/cate_list.json"
if os.path.exists(item_data_fn) and os.path.exists(cate_item_fn) and os.path.exists(item_list_fn):
    print("load files from cache")
    itemData = load_json(item_data_fn)
    itemList = load_json(item_list_fn)
    cateItem = load_json(cate_item_fn)
    cateList = load_json(cate_list_fn)
else:
    itemData = dict()
    itemList = dict()
    cateItem = dict()
    cateList = []
    with open(f"{dataDir}/item_data.txt") as f:
        for line in f:
            item_id, cate_id, pic_url, *title = line.strip().split(",")
            itemData[item_id] = {"cate": cate_id, "url": pic_url, "title": ",".join(title)}
            if cate_id not in cateItem:
                cateItem[cate_id] = set()
            cateItem[cate_id].add(item_id)
    # convert to list
    for cate_id, items in cateItem.items():
        cateItem[cate_id] = list(items)
    itemList = sorted(list(itemData.keys()))
    cateList = sorted(list(cateItem.keys()))
    save_json(item_data_fn, itemData)
    save_json(item_list_fn, itemList)
    save_json(cate_item_fn, cateItem)
    save_json(cate_list_fn, cateList)

num_used_item = np.sort([len(items) for items in cateItem.values()])[::-1]
num_cates = len(cateList)
fig = plt.figure()
plt.plot(num_used_item)
plt.xlabel("Category")
plt.ylabel("Number of items")
plt.title("Number of items in each category")
plt.savefig(f"{outputDir}/category_size.pdf")

print("{:=^25}".format("Item Data"))
print("Number of items: {:,}".format(len(itemData)))
print("Number of categories: {:,}".format(len(cateItem)))

# ========Item Data========
# Number of items: 4,747,039
# Number of categories: 75

# %%


def scan_files(
    path: str = "./", suffix: Union[str, tuple[str]] = "", recursive: bool = False, relpath: bool = False
) -> List:
    """Scan files under path which follows the PEP 471.

    Args:
        path (str, optional): target path. Defaults to "./".
        suffix (Union[str, Tuple[str]], optional): folder that ends with given suffix, it can also be a tuple. Defaults to "".
        recursive (bool, optional): scan files recursively. Defaults to False.
        relpath (bool, optional): return relative path. Defaults to False.

    Returns:
        List: list of files

    """

    def scantree(path):
        for entry in os.scandir(path):
            if not entry.name.startswith("."):
                if entry.is_dir(follow_symlinks=False):
                    yield from scantree(entry.path)
                else:
                    yield entry

    def scandir(path):
        for entry in os.scandir(path):
            if not entry.name.startswith(".") and entry.is_file():
                yield entry

    files = []
    scan = scantree if recursive else scandir
    for entry in scan(path):
        if entry.name.endswith(suffix):
            files.append(os.path.relpath(entry.path, path) if relpath else entry.path)
    return files


download_images = scan_files(imageDir, suffix=("jpg", "jpeg", "png"))
print(f"Number of images: {len(download_images)}")

# %% Process the `outfit_data.txt` file
"""
The outfit_data.txt saves each outfit in the following format:

    outfit_id,item_id;item_id;...

We will convert this file into the following:
- outfitData: {outfit_id: [item_id, item_id, ...]}
- outfitList: [outfit_id]
"""

outfit_data_fn = f"{outputDir}/json/outfit_data.json"
outfit_list_fn = f"{outputDir}/json/outfit_list.json"
if os.path.exists(outfit_data_fn) and os.path.exists(outfit_list_fn):
    outfitData = load_json(outfit_data_fn)
    outfitList = load_json(outfit_list_fn)
else:
    outfitData = dict()
    with open(f"{dataDir}/outfit_data.txt") as f:
        for line in f:
            outfit_id, items = line.strip().split(",")
            outfitData[outfit_id] = items.split(";")
    outfitList = sorted(outfitData.keys())
    save_json(outfit_data_fn, outfitData)
    save_json(outfit_list_fn, outfitList)


num_used_item = []
used_item_set = list()
for items in outfitData.values():
    used_item_set += items
    num_used_item.append(len(items))
used_item_set = set(used_item_set)

fig = plt.figure()
plt.hist(num_used_item, density=True)
plt.xlabel("Outfit length")
plt.ylabel("Density")
plt.title("Outfit length distribution")
plt.savefig(f"{outputDir}/histogram_outfit_length.pdf")

print("{:=^25}".format("Outfit Data"))
print("Number of outfits: {:,}".format(len(outfitData)))
print("Number of items: {:,} [{} - {}]".format(sum(num_used_item), min(num_used_item), max(num_used_item)))
print("Number of unique items: {:,}".format(len(used_item_set)))
print("Ratio of reuse: {:.3f}".format(sum(num_used_item) / len(used_item_set)))
print("Number of items/outfit: {:.3f}".format(sum(num_used_item) / len(outfitData)))

# =======Outfit Data=======
# Number of outfits: 1,013,136
# Number of items: 3,692,882 [3 - 9]
# Number of unique items: 583,464
# Ratio of reuse: 6.329
# Number of items/outfit: 3.645

# %% Process user_data.txt
"""
The `user_data.txt` file saves each user in the following format:

    user_id,item_id;item_id;...,outfit_id

where the item list is the browsing history of the user and the outfit_id is the outfit the user has created.

We convert this file into the following:

- userClick: {user_id: [[item_id, item_id, ...], ...]}
- userOutfit: {user_id: [outfit_id, outfit_id, ...]}
- userList: [user_id]


"""
user_list_fn = f"{outputDir}/json/user_list.json"
user_click_fn = f"{outputDir}/json/user_click.json"
user_outfit_fn = f"{outputDir}/json/user_outfit.json"
if os.path.exists(user_outfit_fn) and os.path.exists(user_list_fn):
    userClick = load_json(user_click_fn)
    userList = load_json(user_list_fn)
    userOutfit = load_json(user_outfit_fn)
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
    save_json(user_click_fn, userClick)
    save_json(user_outfit_fn, userOutfit)
    save_json(user_list_fn, userList)

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
plt.title("Number of outfits")
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

# %% Save the data
used_items = set()
for items in outfitData.values():
    used_items.update(items)
used_items = list(set(used_items))
used_items.sort()
save_json(f"{outputDir}/json/used_items.json", used_items)
print(f"Number of used items: {len(used_items)}")
# %%
"""
We save the outfit data for recommendation task.

In this section, we use the following data:

- userOutfit: {user_id: [outfit_id, outfit_id, ...]}
- outfitData: {outfit_id: [item_id, item_id, ...]}
- itemData: {item_id: {"cate": cate_id, "url": pic_url, "title": title}}

"""
tupleDir = os.path.join(outputDir, "tuple")
os.makedirs(tupleDir, exist_ok=True)
min_num_outfit = 100
validUserOutfit = {k: v for k, v in userOutfit.items() if len(v) >= min_num_outfit}
validUserList = sorted(list(validUserOutfit.keys()))
num_valid_outfits = sum([len(v) for v in validUserOutfit.values()])
num_valid_users = len(validUserList)
print(f"Number of users with at least {min_num_outfit} outfits: {len(validUserOutfit)}")
# %%
user_index = {v: i for i, v in enumerate(validUserList)}
cate_index = {v: i for i, v in enumerate(cateList)}
validItemList = [set() for _ in range(num_cates)]
min_num_item = np.inf
max_num_item = 0
for user_id, outfits in tqdm(validUserOutfit.items(), desc="Counting items"):
    for outfit_id in outfits:
        items = outfitData[outfit_id]
        min_num_item = min(min_num_item, len(items))
        max_num_item = max(max_num_item, len(items))
        for item_id in items:
            item_cate = itemData[item_id]["cate"]
            validItemList[cate_index[item_cate]].add(item_id)
validItemList = [sorted(list(items)) for items in validItemList]
num_valid_items = sum([len(items) for items in validItemList])
print(f"Number of valid users: {len(validUserOutfit):,}")
print(f"Number of valid cates: {len(cate_index):,}")
print(f"Number of valid items: {num_valid_items:,}")
print(f"Min number of items in outfit: {min_num_item}")
print(f"Max number of items in outfit: {max_num_item}")
# save item list
save_json(f"{tupleDir}/items.json", validItemList)
# %%
cate_item_index = [dict() for _ in range(num_cates)]
for i, items in enumerate(validItemList):
    cate_item_index[i] = {v: i for i, v in enumerate(items)}

user_tuples = [[] for _ in range(num_valid_users)]
for i, (user_id, outfits) in enumerate(validUserOutfit.items()):
    for outfit_id in outfits:
        outfit = outfitData[outfit_id]
        user = user_index[user_id]
        size = len(outfit)
        cates = [cate_index[itemData[i]["cate"]] for i in outfit] + [-1] * (max_num_item - size)
        items = [cate_item_index[c][key] for c, key in zip(cates, outfit)] + [-1] * (max_num_item - size)
        user_tuples[user].append([user, size] + items + cates)

# %% Split the data
train_ratio = 0.8
test_ratio = 0.1
valid_ratio = 0.1
train_pos = []
test_pos = []
valid_pos = []
for tuples in user_tuples:
    num_train = int(len(tuples) * train_ratio)
    num_test = int(len(tuples) * test_ratio)
    num_valid = len(tuples) - num_train - num_test
    train_pos.extend(tuples[:num_train])
    test_pos.extend(tuples[num_train : num_train + num_test])
    valid_pos.extend(tuples[num_train + num_test :])
train_pos = np.array(train_pos)
test_pos = np.array(test_pos)
valid_pos = np.array(valid_pos)
save_csv(f"{tupleDir}/train_pos", train_pos)
save_csv(f"{tupleDir}/test_pos", test_pos)
save_csv(f"{tupleDir}/valid_pos", valid_pos)

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
        cate_id = [cate_index[itemData[i]["cate"]] for i in outfit] + append
        tuple_user_item.append([[uidx, size] + item + cate_id])
        tuple_user_outfit.append([uidx, outfit_index[outfit_id]])
    clicks = userClick[user]
    for seq in clicks:
        tuple_user_click.append([uidx] + [item_index[i] for i in seq])

save_csv(f"{outputDir}/tuple_user_item.csv", tuple_user_item)
save_csv(f"{outputDir}/tuple_user_outfit.csv", tuple_user_outfit)
save_csv(f"{outputDir}/tuple_user_click.csv", tuple_user_click)
