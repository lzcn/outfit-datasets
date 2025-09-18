## Data Format

The [iFashion dataset](https://github.com/wenyuer/POG) originally provides only the following three files:

### Outfit data

The `outfit_data.txt` saves a list of outfits with following format:
```
outfit_id,item_id;item_id;item_id...
```

### User data

The `user_data.txt` saves a list of user, item history, outfit pairs with following format:
```
user_id,item_id;item_id;...,outfit_id
```
 
### Item data

The `item_data.txt` saves a list of items with following format:
```
item_id,cate_id,pic_url,title
```
