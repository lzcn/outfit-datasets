#!/usr/bin/env python
import argparse
import json
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
    def __init__(self, env, msg="Default debugger"):
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


class PolyvoreImage(Dataset):
    def __init__(self, image_keys, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.image_keys = image_keys
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        key = self.image_keys[index]
        path = os.path.join(f"{self.data_dir}", "images", f"{key}.jpg")
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return key, self.transform(img)

    def __len__(self):
        return len(self.image_keys)


@torch.no_grad()
def main(args):
    all_items = torchutils.io.load_json(f"{args.data_dir}/polyvore_item_metadata.json")
    image_keys = list(all_items.keys())
    lmdb_dir = os.path.join(args.feature_dir, args.backbone)
    os.makedirs(lmdb_dir, exist_ok=True)
    print("Getting dataloader.")
    loader = DataLoader(dataset=PolyvoreImage(image_keys, args.data_dir), batch_size=64, num_workers=8, shuffle=False,)
    print("Getting backbone: {}.".format(args.backbone))
    model, _ = torchutils.backbone(args.backbone)
    model.cuda()
    model.eval()
    env = lmdb.open(lmdb_dir, map_size=2 ** 40)
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
        description="Extract feature from given backbone.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="release")
    parser.add_argument("--feature-dir", default="processed/features")
    parser.add_argument("--backbone", default="alexnet", help="Backbone")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    main(args)
