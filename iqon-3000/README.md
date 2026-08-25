# IQON 3000

Download the raw data from [GP-BPR](https://github.com/hanxjing/GP-BPR/)

```text
raw/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
├── feat/
│   ├── smallnwjc2vec
│   ├── textfeatures
│   └── visualfeatures
└── images/ (IQON3000.zip)
    └── user_id/
        └── outfit_id/
            ├── outfit_id.json
            ├── item_id.jpg
            └── ...
```

1. `merge_json.py`: Merge the json files in the `images` folder to a single json file `processed/outfits.json`.

2. `extract_features.py`: Extract the visual features from the images and save them to `processed/features`.

{'accessories': 0, 'bag': 1, 'bottom': 2, 'coat': 3, 'dress': 4, 'hat': 5, 'shoes': 6, 'top': 7}
