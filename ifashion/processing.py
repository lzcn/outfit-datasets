# %%
import os
import torchutils
import json
import pprint
import tqdm
import requests

# %%

data_dir = "raw"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

# %%

# load item data
item_cate = dict()
cate_set = set()
item_set = set()
with open("raw/item_data.txt") as f:
    for line in f:
        line = line.strip().split(",")
        item_id, cate = line[0], line[1]
        item_set.add(item_id)
        cate_set.add(cate)
        if cate not in item_cate:
            item_cate[cate] = set()
        item_cate[cate].add(item_id)

for k, v in item_cate.items():
    item_cate[k] = list(v)
torchutils.io.save_json(f"{output_dir}/item_cate.json", item_cate)
print("{:=^25}".format("Item Data"))
print("Number of items: {:,}".format(len(item_set)))
print("Number of categories: {:,}".format(len(cate_set)))

# ========Item Data========
# Number of items: 4,747,039
# Number of categories: 75

# %%

user_item_set = []
user_outfit_set = set()
user_set = set()
with open("raw/user_data.txt") as f:
    for line in f:
        user, item_ids, outfit = line.strip().split(",")
        items = item_ids.split(";")
        user_item_set += items
        user_set.add(user)
        user_outfit_set.add(outfit)
user_item_set = set(user_item_set)

print("{:=^25}".format("User Data"))
print("Number of outfits: {:,}".format(len(user_outfit_set)))
print("Number of items: {:,}".format(len(user_item_set)))
print("Number of users: {:,}".format(len(user_set)))

# ========User Data========
# Number of outfits: 127,169
# Number of items: 4,463,302
# Number of users: 3,569,112

# %%

outfit_set = set()
item_set = []
with open("raw/outfit_data.txt") as f:
    for line in f:
        outfit_id, items = line.strip().split(",")
        item_set += items.split(";")
        outfit_set.add(outfit_id)
item_set = set(item_set)

print("{:=^25}".format("Outfit Data"))
print("Number of outfits: {:,}".format(len(outfit_set)))
print("Number of items: {:,}".format(len(item_set)))

# =======Outfit Data=======
# Number of outfits: 1,013,136
# Number of items: 583,464

# %%
