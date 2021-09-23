#!/usr/bin/env python
import argparse
import json
import os
import time
from multiprocessing import Pool
import random
import requests
import torchutils
from tqdm.auto import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}


def download_image(img_path, img_link):
    if os.path.exists(img_path):
        return
    while True:
        try:
            time.sleep(random.random() / 10)
            response = requests.get(img_link, headers=headers, timeout=5)
            if response.status_code == 200:
                file = open(img_path, "wb")
                file.write(response.content)
                file.close()
            else:
                with open("failed.txt", "a") as f:
                    f.write(f"{img_link}\n")
                tqdm.write(f"Failed. {img_link}")
            break
        except Exception as e:
            tqdm.write("Exception: {}. {}".format(e, img_link))
            time.sleep(random.random() * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all JSON file", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--txt-file", default="raw/item_data.txt")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--num-workers", default=16, type=int)
    args = parser.parse_args()
    # load urls
    print("Reading urls")
    if os.path.exists("failed.txt"):
        with open("failed.txt") as f:
            broken_url = f.read()
            broken_url = broken_url.split("\n")
    else:
        broken_url = []
    broken_url = set(broken_url)
    # downloaded = set(torchutils.files.scan_files(args.image_dir, recursive=True))
    todownload = []
    item_set = set()
    with open(args.txt_file) as f:
        for line in f:
            item_id, _, url, *title = line.strip().split(",")
            if item_id in item_set:
                continue
            else:
                item_set.add(item_id)
            if "http:" not in url:
                url = "http://" + url.split("//")[-1]
            suffix = url.split(".")[-1].lower()
            if suffix not in ["jpg", "png", "jpeg"]:
                suffix = "jpg"
            img_path = os.path.join(args.image_dir, f"{item_id}.{suffix}")
            if url in broken_url:
                continue
            todownload.append((img_path, url))
    print("Number of all items: {:,}".format(len(todownload) + len(broken_url)))
    print("Number of items to download: {:,}".format(len(todownload)))
    # print("Number of items downloaded: {:,}".format(len(downloaded)))
    print("Number of broken items: {:,}".format(len(broken_url)))

    pbar = tqdm(total=len(todownload))

    def update(*a):
        pbar.update()

    tqdm.write("Adding tasks to pool.")
    pool = Pool(args.num_workers)
    for img_path, img_url in todownload:
        pool.apply_async(download_image, args=(img_path, img_url), callback=update)
    tqdm.write("All tasks are scheduled.")
    pool.close()
    pool.join()
    tqdm.write("Finished. Please delete cache file for broken urls and re-run to check!")
    exit(0)
