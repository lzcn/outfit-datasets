# %%
import os

import torchutils

# %%
inputDir = "release"
outputDir = "processed"
os.makedirs(f"{outputDir}/hfgn", exist_ok=True)


def read_image_list(fn):
    with open(fn) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


imageList = [
    read_image_list(os.path.join(inputDir, "img_list", f"img_list_{cate}.txt")) for cate in ["top", "bottom", "shoe"]
]


torchutils.io.save_json(f"{outputDir}/hfgn/items.json", imageList)
# %%


def read_index_map(fn):
    # index map to image list
    with open(fn) as f:
        lines = [int(line.strip()) for line in f.readlines()]
    return lines


# map to the item index in image list
trainIndexMapImage = [
    read_index_map(os.path.join(inputDir, "tuple_180", f"{cate}_ind_train.txt")) for cate in ["top", "bottom", "shoe"]
]
testIndexMapImage = [
    read_index_map(os.path.join(inputDir, "tuple_180", f"{cate}_ind_test.txt")) for cate in ["top", "bottom", "shoe"]
]

item_set = [set(), set(), set()]
for c, items in enumerate(trainIndexMapImage):
    for item in items:
        item_set[c].add(item)

for c, items in enumerate(testIndexMapImage):
    for item in items:
        item_set[c].add(item)

item_index_map = [dict(), dict(), dict()]
for c, items in enumerate(item_set):
    items = sorted(list(items))
    for n, item in enumerate(items):
        item_index_map[c][item] = n

print("Statistics:")
for n, image_list in enumerate(imageList):
    print("Number of images for {}-th cate: {:,}".format(n, len(image_list)))

for n, items in enumerate(trainIndexMapImage):
    print("Number of items (train) {}-th cate: {:,}".format(n, len(items)))

for n, items in enumerate(testIndexMapImage):
    print("Number of items (train) {}-th cate: {:,}".format(n, len(items)))

for n, items in enumerate(item_set):
    print("Number of items (all) {}-th cate: {:,}".format(n, len(items)))

# %%


def read_tuples(fn):
    with open(fn) as f:
        lines = [list(map(int, line.strip().split())) for line in f.readlines()]
    return lines


train_pos = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_train_posi.txt"))
train_neg = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_train_neg.txt"))
test_pos = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_test_posi.txt"))
test_neg = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_test_neg.txt"))

print("Number of outfits: {:,}".format(len(train_pos) + len(train_neg) + len(test_neg) + len(test_pos)))

# %%


def remap_outfits(outfits, index_map):
    remaped_outfits = []
    for outfit in outfits:
        uidx, *items = outfit
        items = [index_map[n][idx] for n, idx in enumerate(items)]
        remapped_items = [item_index_map[n][idx] for n, idx in enumerate(items)]
        remaped_outfits.append((uidx, tuple(remapped_items)))
    return remaped_outfits


remapped_train_pos = remap_outfits(train_pos, trainIndexMapImage)
remapped_train_neg = remap_outfits(train_neg, trainIndexMapImage)

remapped_test_pos = remap_outfits(test_pos, testIndexMapImage)
remapped_test_neg = remap_outfits(test_neg, testIndexMapImage)

all_outfits = set()
n_outfits = 0

for uix, outfit in remapped_train_pos:
    all_outfits.add(outfit)
    n_outfits += 1

outfit_index_map = dict()
for n, outfit in enumerate(all_outfits):
    outfit_index_map[outfit] = n

print("Number of outfits:", len(all_outfits), "Number of used times:", n_outfits)


# %%

# create user-outfit tuples
def get_uo(outfits):
    pass


# %%


def convert_tuples(tuples, index_map):
    conveted = []
    for outfit in tuples:
        uidx, *items = outfit
        items = [index_map[n][idx] for n, idx in enumerate(items)]
        types = [0, 1, 2]
        size = 3
        conveted.append([uidx, size] + items + types)
    return conveted


train_pos_converted = convert_tuples(train_pos, trainIndexMapImage)
train_neg_converted = convert_tuples(train_neg, trainIndexMapImage)
test_pos_converted = convert_tuples(test_pos, testIndexMapImage)
test_neg_converted = convert_tuples(test_neg, testIndexMapImage)

torchutils.io.save_csv(f"{outputDir}/hfgn/train_pos", train_pos_converted)
torchutils.io.save_csv(f"{outputDir}/hfgn/train_neg", train_neg_converted)
torchutils.io.save_csv(f"{outputDir}/hfgn/test_pos", test_pos_converted)
torchutils.io.save_csv(f"{outputDir}/hfgn/test_neg", test_neg_converted)
