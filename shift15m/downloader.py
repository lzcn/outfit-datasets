#!/usr/bin/env python
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import requests
import torchutils
from tqdm.auto import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}


def get_link(path):
    # example: https://img.iqon.jp/items/34255717/34255717_l.jpg
    name = os.path.basename(path)
    return os.path.join("https://img.iqon.jp/items/", name.split("_")[0], name)


def download_image(img_path):
    img_url = get_link(img_path)
    while True:
        try:
            response = requests.get(img_url, headers=headers, timeout=5)
            if response.status_code == 200:
                file = open(img_path, "wb")
                file.write(response.content)
                file.close()
            else:
                tqdm.write("Failed. {}".format(img_url))
            break
        except Exception as e:
            tqdm.write("Exception: {}".format(e))
            time.sleep(random.random() * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-file", default="release/iqon_outfits.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--image-size", default="all")
    parser.add_argument("--num-workers", default=16, type=int)
    args = parser.parse_args()
    folder = args.image_dir
    print("Loading JSON file for all items")
    with open(args.json_file, "r") as f:
        outfits = json.load(f)
    if args.image_size == "small":
        suffixs = ("_s.jpg",)
    elif args.image_size == "middle":
        suffixs = ("_m.jpg",)
    elif args.image_size == "large":
        suffixs = ("_l.jpg",)
    else:
        suffixs = ("_s.jpg", "_m.jpg", "_l.jpg")
    todownload = []
    for outfit in outfits:
        set_id = str(outfit["set_id"])
        user_id = str(outfit["user"]["user_id"])
        for item in outfit["items"]:
            item_id = str(item["item_id"])
            for suffix in suffixs:
                # os.makedirs(os.path.join(folder, user_id, set_id), exist_ok=True)
                todownload.append(os.path.join(folder, user_id, set_id, item_id + suffix))
    downloaded = set(torchutils.files.scan_files(folder, suffixs, recursive=True))
    print("Number of items to download: {:,}".format(len(todownload)))
    print("Number of items downloaded: {:,}".format(len(downloaded)))

    pbar = tqdm(total=len(todownload))

    def update(*a):
        pbar.update()

    tqdm.write("Adding tasks to pool.")
    pool = Pool(args.num_workers)
    for img_path in todownload:
        if img_path in downloaded:
            pbar.update()
        else:
            pool.apply_async(download_image, args=(img_path,), callback=update)
    tqdm.write("All tasks are scheduled.")
    pool.close()
    pool.join()
    tqdm.write("Finished.")
    exit(0)
