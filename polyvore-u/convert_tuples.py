# %%
import os

import numpy as np
import pandas as pd
import torchutils

# %%
inputDir = "polyvore-u/release"
outputDir = "polyvore-u/processed"
dataSet = "tuples_630"
saveFile = False

os.makedirs(f"{outputDir}/original/{dataSet}", exist_ok=True)


# %%
def load_image_list(fn):
    with open(fn) as f:
        lines = f.read().strip().split()
    return lines


image_list = [
    load_image_list(os.path.join(f"{inputDir}/{dataSet}/image_list_top")),
    load_image_list(os.path.join(f"{inputDir}/{dataSet}/image_list_bottom")),
    load_image_list(os.path.join(f"{inputDir}/{dataSet}/image_list_shoe")),
]
torchutils.io.save_json(f"{outputDir}/original/{dataSet}/items.json", image_list)

# %%
imageReader = torchutils.data.ImageLMDBReader(f"{outputDir}/features/images")

# %%
image = np.array(imageReader(image_list[0][0]))
print("Image shape: {}".format(image.shape))


# %%
def rearrange(items, types):
    new_items, new_types = [], []
    for item_id, item_type in zip(items, types):
        if item_type == -1:
            continue
        new_items.append(item_id)
        new_types.append(item_type)
    while len(new_items) < len(items):
        new_items.append(-1)
        new_types.append(-1)
    return new_items + new_types


def convert_tuples(data: np.ndarray):
    uidx, tuples = data[:, 0], data[:, 1:]
    n, m = tuples.shape
    if m == 3:
        types = np.array([0, 1, 2]).reshape((1, -1)).repeat(n, axis=0)
    else:
        types = np.array([0, 0, 1, 2]).reshape((1, -1)).repeat(n, axis=0)
    types = np.where(tuples == -1, tuples, types)
    size = np.sum(types != -1, axis=1)
    converted = []
    for i in range(n):
        converted.append([uidx[i], size[i]] + rearrange(tuples[i], types[i]))
    return np.array(converted)


# %%
data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_train_posi", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/train_pos", converted)

data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_train_nega", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/train_neg", converted)


# %%
data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_val_posi", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/valid_pos", converted)

data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_val_nega", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/valid_neg", converted)


# %%
data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_test_posi", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/test_pos", converted)

data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/tuples_test_nega", dtype=np.int64))
converted = convert_tuples(data)
if saveFile:
    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/test_neg", converted)


# %%
data = np.array(pd.read_csv(f"{inputDir}/{dataSet}/fill_in_blank_test", dtype=np.int64))
num_answers = 4
num_questions = data.shape[0]
num_columns = data.shape[1] // num_answers
pos = data[:, :num_columns]
neg = data[:, num_columns:].reshape((num_questions * (num_answers - 1), num_columns))
pos = convert_tuples(pos)
neg = convert_tuples(neg)

data = data.reshape((num_questions * num_answers, -1))
converted = convert_tuples(data)
torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/test_fitb", converted)
torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/test_pos_fitb", pos)
torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/test_neg_fitb", neg)


# %%
if dataSet == "tuples_630":
    retrieval_outfits = []
    retrieval_top = np.array(pd.read_csv(f"{inputDir}/{dataSet}/retrieval_top_all", dtype=np.int64))
    num_outfits, num_items = retrieval_top.shape
    num_items = num_items // 4
    print("Retrieval Top item:", retrieval_top.shape)
    converted_top = retrieval_top.reshape((num_outfits, num_items, -1))
    outfits_top = converted_top[:, 0, :]
    outfits_top = convert_tuples(outfits_top)

    retrieval_bottom = np.array(pd.read_csv(f"{inputDir}/{dataSet}/retrieval_bottom_all", dtype=np.int64))
    num_outfits, num_items = retrieval_bottom.shape
    num_items = num_items // 4
    print("Retrieval Bottom item:", retrieval_bottom.shape)
    converted_bottom = retrieval_bottom.reshape((num_outfits, num_items, -1))
    outfits_bottom = converted_bottom[:, 0, :]
    outfits_bottom = convert_tuples(outfits_bottom)

    retrieval_shoe = np.array(pd.read_csv(f"{inputDir}/{dataSet}/retrieval_shoe_all", dtype=np.int64))
    num_outfits, num_items = retrieval_shoe.shape
    num_items = num_items // 4
    print("Retrieval Shoe item:", retrieval_shoe.shape)
    converted_shoe = retrieval_shoe.reshape((num_outfits, num_items, -1))
    outfits_shoe = converted_shoe[:, 0, :]
    outfits_shoe = convert_tuples(outfits_shoe)

    print("Top    outfits:", outfits_top.shape)
    print("Bottom outfits:", outfits_bottom.shape)
    print("Shoe   outfits:", outfits_shoe.shape)

    print("Top == Bottom:", (outfits_top == outfits_bottom).all())
    print("Top == Shoe:", (outfits_top == outfits_shoe).all())

    torchutils.io.save_csv(f"{outputDir}/original/{dataSet}/retrieval", outfits_top)
