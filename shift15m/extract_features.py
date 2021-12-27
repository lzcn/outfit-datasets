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


class IQONImage(Dataset):
    def __init__(self, image_dir, image_size="middle"):
        super().__init__()
        suffix = dict(small="s.jpg", middle="m.jpg", large="l.jpg")[image_size]
        self.image_dir = image_dir
        self.image_list = torchutils.files.scan_files(image_dir, suffix, recursive=True)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def __getitem__(self, index):
        path = self.image_list[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        name = os.path.basename(path).split("_")[0]
        return name, self.transform(img)

    def __len__(self):
        return len(self.image_list)


@torch.no_grad()
def main(args):
    backbone = args.backbone
    lmdb_dir = os.path.join(args.feature_dir, backbone)
    os.makedirs(lmdb_dir, exist_ok=True)
    print("Getting dataloader.")
    loader = DataLoader(
        dataset=IQONImage(args.image_dir), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )
    print("Getting backbone: {}.".format(backbone))
    model, _ = torchutils.backbone(backbone)
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
    parser.add_argument("--image-dir")
    parser.add_argument("--feature-dir")
    parser.add_argument("--backbone", default="alexnet", help="Backbone")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    main(args)
