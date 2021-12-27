# %%
import datetime
import gzip
import io
import json
import os
import pprint
from collections import defaultdict

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torchutils
from tqdm import tqdm

# %% [markdown]

# ## Dataset structure
#
# The original dataset is maintained in json format, and a row consists of the following:
# ```json
# {
#   "user":{"user_id":"xxxx", "fav_brand_ids":"xxxx,xx,..."},
#   "like_num":"xx",
#   "set_id":"xxx",
#   "items":[
#     {"price":"xxxx","item_id":"xxxxxx","category_id1":"xx","category_id2":"xxxxx"},
#     ...
#   ],
#   "publish_date":"yyyy-mm-dd"
# }
# ```

# %%

outputDir = "processed"
inputDir = "release"
# %%

outfits = torchutils.io.load_json(f"{inputDir}/iqon_outfits.json")
cate = [[i["category_id1"] for i in t["items"]] for t in outfits]
like = [int(t["like_num"]) for t in outfits]
date = [datetime.datetime.strptime(t["publish_date"], "%Y-%m-%d") for t in outfits]
# %%
pprint.pprint(outfits[0])
# %%

user_fav_brands = dict()
for outfit in tqdm(outfits, "user_fav_brands"):
    user_id = outfit["user"]["user_id"]
    if user_id not in user_fav_brands:
        user_fav_brands[user_id] = set()
    fav_brand_ids = outfit["user"].get("fav_brand_ids", None)
    if fav_brand_ids is not None:
        fav_brand_ids = fav_brand_ids.split(",")
        user_fav_brands[user_id].update(fav_brand_ids)
user_fav_brands = {user_id: list(brands) for user_id, brands in user_fav_brands.items()}
torchutils.io.save_json(f"{outputDir}/user_fav_brands.json", user_fav_brands, overwrite=True)
# %%

item_meta_data = dict()
item_dict_1 = defaultdict(set)
item_dict_2 = defaultdict(set)
for outfit in tqdm(outfits, "item_meta_data"):
    for item in outfit["items"]:
        item_id = item["item_id"]
        price = item["price"]
        category_id1 = item["category_id1"]
        category_id2 = item["category_id2"]
        item_meta_data[item_id] = {
            "item_id": item_id,
            "price": price,
            "category_id1": category_id1,
            "category_id2": category_id2,
        }
        item_dict_1[category_id1].add(item_id)
        item_dict_2[category_id2].add(item_id)
torchutils.io.save_json(f"{outputDir}/item_meta_data.json", item_meta_data, overwrite=True)
cate_map_1 = {cate_id: i for i, cate_id in enumerate(sorted(item_dict_1.keys()))}
cate_map_2 = {cate_id: i for i, cate_id in enumerate(sorted(item_dict_2.keys()))}
print("Number of items: {:,}".format(len(item_meta_data)))
print("Number of category: {:,}\n{}".format(len(cate_map_1), cate_map_1))
print("Number of category: {:,}\n{}".format(len(cate_map_2), cate_map_2))

# %%
files = torchutils.files.scan_files(os.path.join(inputDir, "features"))
featureDir = os.path.join(outputDir, "features/vgg")
os.makedirs(featureDir, exist_ok=True)
with lmdb.open(os.path.join(outputDir, "features/vgg"), map_size=2 ** 40) as env:
    with env.begin(write=True) as txn:
        for fn in tqdm(files, desc="Extract features"):
            key = fn.split("/")[-1].split(".")[0]
            with gzip.open(fn, "r") as f:
                outfits = f.read()
                feat = json.loads(outfits.decode("utf-8"))
                feat = np.array(feat, np.float32)
            txn.put(key.encode(), feat)
# %%

cleaned_data = []
for outfit in tqdm(outfits, "Clean outfits"):
    cleaned_data.append(
        {
            "user_id": outfit["user"]["user_id"],
            "set_id": outfit["set_id"],
            "items": [i["item_id"] for i in outfit["items"]],
            "cate1": [i["category_id1"] for i in outfit["items"]],
            "cate2": [i["category_id2"] for i in outfit["items"]],
            "likes": outfit["like_num"],
            "date": outfit["publish_date"],
        }
    )
torchutils.io.save_json(f"{outputDir}/outfits.json", cleaned_data, overwrite=True)
print("Number of items: {:,}".format(len(cleaned_data)))

# %%

# https://github.com/st-tech/zozo-shift15m/blob/main/benchmarks/set_matching/outfits/split_trainval.py


years = [
    [2013, 2013],
    [2013, 2014],
    [2013, 2015],
    [2013, 2016],
    [2013, 2017],
]

train = 30816
val = 3851
test = 3851

output_root = "processed"
month = 1
day = 1
hour = 0
minute = 0


def one_year_interval(year, month=1, day=1, hour=0, minute=0):
    start = datetime.datetime(year, month, day, hour, minute)
    end = datetime.datetime(year + 1, month, day, hour, minute)
    return start, end


def save_outfits(data_x, data_y, year_x, year_y, output_dir, num_train=30816, num_valid=3851, num_test=3851, seed=0):
    np.random.seed(seed)
    data_perm = np.random.permutation(data_x)
    data_tr = data_perm[0:num_train]
    if year_x != year_y:
        np.random.seed(0)
        data_perm = np.random.permutation(data_y)
        data_vl = data_perm[0:num_valid]
        data_te = data_perm[num_valid : num_valid + num_test]
    else:
        data_vl = data_perm[num_train : num_train + num_valid]
        data_te = data_perm[num_train + num_valid : num_train + num_valid + num_test]
    data_tr = data_tr.tolist()
    data_vl = data_vl.tolist()
    data_te = data_te.tolist()
    output_dir = os.path.join(output_root, f"{year_x}-{year_y}-seed-{seed}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(data_tr, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "valid.json"), "w") as f:
        json.dump(data_vl, f, indent=4, ensure_ascii=True)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(data_te, f, indent=4, ensure_ascii=True)


for (y_i, y_j) in years:
    date_i_start, date_i_end = one_year_interval(y_i, 1, 1, 0, 0)
    date_j_start, date_j_end = one_year_interval(y_j, 1, 1, 0, 0)
    ind_i = [(date_i_start <= d < date_i_end) and (l >= 50) and (len(set(c)) >= 4) for d, l, c in zip(date, like, cate)]
    ind_j = [(date_j_start <= d < date_j_end) and (l >= 50) and (len(set(c)) >= 4) for d, l, c in zip(date, like, cate)]
    data_i = np.array(cleaned_data)[ind_i]
    data_j = np.array(cleaned_data)[ind_j]

    save_outfits(data_i, data_j, y_i, y_j, output_dir=output_root, seed=0)
    save_outfits(data_i, data_j, y_i, y_j, output_dir=output_root, seed=1)
    save_outfits(data_i, data_j, y_i, y_j, output_dir=output_root, seed=2)


# %%
