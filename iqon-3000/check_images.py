#!/usr/bin/env python
import argparse
import os
import time
from multiprocessing import Pool

import requests
import torchutils
from PIL import Image
from tqdm.auto import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"
}


def get_link(path):
    # get image link from name
    # example: https://img.iqon.jp/items/34255717/34255717_m.jpg
    name = os.path.basename(path)
    return os.path.join("https://img.iqon.jp/items/", name.split("_")[0], name)


def check_image(img_path):
    """Check wether an image is broken and re-download when it is broken"""
    try:
        with open(img_path, "rb") as f:
            Image.open(f).convert("RGB")
    except Exception as e:
        tqdm.write("Error {} for image: {}".format(e, img_path))
        img_link = get_link(img_path)
        while True:
            try:
                response = requests.get(img_link, headers=headers, timeout=5)
                if response.status_code == 200:
                    file = open(img_path, "wb")
                    file.write(response.content)
                    file.close()
                    break
                else:
                    tqdm.write("Download Failed.")
                    time.sleep(60)
            except Exception as e:
                tqdm.write("Exception: {}".format(e))
                time.sleep(60)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check download images.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", default="raw/images")
    parser.add_argument("--image-size", default="middle")
    parser.add_argument("--num-workers", default=1, type=int)
    args = parser.parse_args()
    folder = args.image_dir
    if args.image_size == "small":
        suffixs = ("_s.jpg",)
    elif args.image_size == "middle":
        suffixs = ("_m.jpg",)
    elif args.image_size == "large":
        suffixs = ("_l.jpg",)
    else:
        suffixs = ("_s.jpg", "_m.jpg", "_l.jpg")
    all_images = torchutils.files.scan_files(folder, suffixs, recursive=True)
    total = len(all_images)
    pool = Pool(args.num_workers)
    pbar = tqdm(total=total)

    def update(*a):
        pbar.update()

    for img_path in all_images:
        pool.apply_async(check_image, args=(img_path,), callback=update)
    tqdm.write("Scheduled")
    pool.close()
    pool.join()
    print("All checked.")
