#!/usr/bin/env python
import argparse
import os
import random
import time
from multiprocessing import Pool

import requests
import torchutils
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}


def download_image(img_path, img_link, tmp_file):
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
                with open(tmp_file, "a") as f:
                    f.write(f"{img_link}\n")
                tqdm.write(f"Failed. {img_link}")
            break
        except Exception as e:
            tqdm.write("Exception: {}. {}".format(e, img_link))
            time.sleep(random.random() * 10)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="iFashon image downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-file", default="release/item_data.txt", help="Path to item data file")
    parser.add_argument("--image-dir", default="images", help="Path to save image")
    parser.add_argument("--num-workers", default=16, type=int, help="Number of processers to use")
    # fmt: on
    args = parser.parse_args()
    print("Reading item urls ...")
    tmp_file = "broken_urls.txt"
    if os.path.exists(tmp_file):
        with open(tmp_file) as f:
            broken_url = f.read()
            broken_url = broken_url.split("\n")
    else:
        broken_url = []
    broken_url = set(broken_url)
    downloaded = set(torchutils.files.scan_files(args.image_dir))
    todownload = []
    item_set = set()
    with open(args.data_file) as f:
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
    print("Number of items downloaded: {:,}".format(len(downloaded)))
    print("Number of broken items: {:,}".format(len(broken_url)))

    pbar = tqdm(total=len(todownload))

    def update(*a):
        pbar.update()

    tqdm.write("Adding tasks to pool ...")
    pool = Pool(args.num_workers)
    for img_path, img_url in todownload:
        pool.apply_async(download_image, args=(img_path, img_url, tmp_file), callback=update)
    tqdm.write("All tasks are scheduled.")
    pool.close()
    pool.join()
    tqdm.write("Finished. You can now delete the cache file for broken urls and re-run to check!")
    exit(0)
