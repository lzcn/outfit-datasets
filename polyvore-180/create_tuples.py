# %%
import os

import torchutils

# %%
inputDir = "release"
outputDir = "processed"
os.makedirs(f"{outputDir}/original", exist_ok=True)


def read_image_list(fn):
    with open(fn) as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


imageList = [
    read_image_list(os.path.join(inputDir, "img_list", f"img_list_{cate}.txt")) for cate in ["top", "bottom", "shoe"]
]
torchutils.io.save_json(f"{outputDir}/original/items.json", imageList)
# %%


def read_index_map(fn):
    with open(fn) as f:
        lines = [int(line.strip()) for line in f.readlines()]
    return lines


trainIndexMap = [
    read_index_map(os.path.join(inputDir, "tuple_180", f"{cate}_ind_train.txt")) for cate in ["top", "bottom", "shoe"]
]
testIndexMap = [
    read_index_map(os.path.join(inputDir, "tuple_180", f"{cate}_ind_test.txt")) for cate in ["top", "bottom", "shoe"]
]


# %%


def read_tuples(fn):
    with open(fn) as f:
        lines = [list(map(int, line.strip().split())) for line in f.readlines()]
    return lines


train_pos = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_train_posi.txt"))
train_neg = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_train_neg.txt"))
test_pos = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_test_posi.txt"))
test_neg = read_tuples(os.path.join(inputDir, "tuple_180", "tuples_test_neg.txt"))


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


train_pos_converted = convert_tuples(train_pos, trainIndexMap)
train_neg_converted = convert_tuples(train_neg, trainIndexMap)
test_pos_converted = convert_tuples(test_pos, testIndexMap)
test_neg_converted = convert_tuples(test_neg, testIndexMap)

torchutils.io.save_csv(f"{outputDir}/original/train_pos", train_pos_converted)
torchutils.io.save_csv(f"{outputDir}/original/train_neg", train_neg_converted)
torchutils.io.save_csv(f"{outputDir}/original/test_pos", test_pos_converted)
torchutils.io.save_csv(f"{outputDir}/original/test_neg", test_neg_converted)
