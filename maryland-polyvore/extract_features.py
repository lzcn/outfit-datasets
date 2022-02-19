#!/usr/bin/env python
import argparse
import os
import threading
from queue import Queue

import lmdb
import torch
import torchutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class Extractor(threading.Thread):
    def __init__(self, env):
        threading.Thread.__init__(self)
        self.queue = Queue()
        self.daemon = True
        self.env = env

    def put(self, data):
        self.queue.put(data)

    def run(self):
        print("Starting %s (Extractor)" % threading.currentThread().getName())
        while True:
            data = self.queue.get()
            if data is None:
                return
            names, features = data
            with self.env.begin(write=True) as txn:
                for key, value in zip(names, features):
                    txn.put(key.encode(), value)


class PolyvoreImages(Dataset):
    def __init__(self, image_dict):
        super().__init__()
        self.image_dict = image_dict
        self.image_keys = list(image_dict.keys())

        self.transform = transforms.Compose(
            [
                transforms.Resize((250, 250)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        key = self.image_keys[index]
        path = self.image_dict[key][0]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return key, self.transform(img)

    def __len__(self):
        return len(self.image_keys)


@torch.no_grad()
def main(args):
    trainOutfits = torchutils.io.load_json(os.path.join(args.data_dir, "label/train_no_dup.json"))
    validOutfits = torchutils.io.load_json(os.path.join(args.data_dir, "label/valid_no_dup.json"))
    testOutfits = torchutils.io.load_json(os.path.join(args.data_dir, "label/test_no_dup.json"))
    allOutfits = trainOutfits + validOutfits + testOutfits
    itemDict = dict()
    for outfit in allOutfits:
        set_id = outfit["set_id"]
        for item in outfit["items"]:
            # the unique id of item
            tid = item["image"].split("tid=")[-1]
            if tid not in itemDict:
                itemDict[tid] = []
            itemDict[tid].append(os.path.join(args.data_dir, "images", "{}/{}.jpg".format(set_id, item["index"])))

    backbone = args.backbone
    lmdb_dir = os.path.join(args.feature_dir, backbone)
    os.makedirs(lmdb_dir, exist_ok=True)
    print("Getting dataloader.")
    loader = DataLoader(
        dataset=PolyvoreImages(itemDict), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    print("Getting backbone: {}.".format(backbone))
    model, _ = torchutils.backbone(backbone)
    model.cuda()
    model.eval()
    env = lmdb.open(lmdb_dir, map_size=2**40)
    extractor = Extractor(env)
    extractor.start()
    for names, x in tqdm(loader, desc="Extract features"):
        x = x.cuda()
        x = model(x).data.cpu().numpy()
        extractor.put((names, x))
    extractor.put(None)
    # wait all data are writed into file
    extractor.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract feature from given backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--feature-dir")
    parser.add_argument("--backbone", default="alexnet")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    main(args)
