import os
import json
import numpy as np
import pandas as pd 
import random

np.random.seed(42)
random.seed(42)

def convert_to_train_format(anns, with_label=False):
    # converts to annotations train dataset format
    new_anns = []
    for index in range(len(anns)):
        for caption in anns[index]['caption']:
            if with_label:
                new_anns.append({"image": anns[index]["image"], 
                            "caption": caption, "image_id": anns[index]["image"], 
                            "label": anns[index]["label"]})
            else:
                new_anns.append({"image": anns[index]["image"], 
                            "caption": caption, "image_id": anns[index]["image"]})
    return new_anns

def convert_to_test_format(anns):
    # converts to annotations test dataset format
    new_anns = []
    img_set = set()
    flag=False
    for ann in anns:
        if ann['image'] not in img_set:
            if flag:
                new_anns.append(a)
            img_set.add(ann['image'])
            a = ann
            a['caption'] = []
        a['caption'].append(ann['caption'])
        flag = True
    return new_anns

def pascal_train_preprocess():
    df = pd.read_csv('pascal_train.csv', sep=",")
    df['img_name']=df['img_name'].apply(lambda x: os.path.join('pascal',x.split('/')[1], x.split('/')[2]))
    df.rename(columns={'img_name':'image'}, inplace=True)
    print(df.head(10))
    df['image_id'] = df['image']
    print(df.head(10))

    with open('pascal_train.json', 'w') as f:
        f.write(df.to_json(orient = 'records'))
 
def pascal_test_preprocess():
    df = pd.read_csv('pascal_test.csv', sep=",")
    df['img_name']=df['img_name'].apply(lambda x: os.path.join('pascal',x.split('/')[1], x.split('/')[2]))
    df.rename(columns={'img_name':'image'}, inplace=True)
    df['image_id'] = df['image']
    print(df.head())
    df = df.groupby(['image', 'label'])['caption'].apply(lambda x:list(x)).reset_index()
    print(df.head())

    with open('pascal_test.json', 'w') as f:
        f.write(df.to_json(orient = 'records'))


def coco_preprocess():
    img2supercls, img2cls, img2clsid = get_coco_dict(filename = 'coco/annotations/instances_val2014.json')
    # print("val2014 labeled images: {}".format(len(val_img2clsid.keys())))
    tr_img2supercls, tr_img2cls, tr_img2clsid = get_coco_dict(filename = 'coco/annotations/instances_train2014.json')
    # print("train2014 labeled images: {}".format(len(tr_img2clsid.keys())))
    img2supercls.update(tr_img2supercls)
    img2cls.update(tr_img2cls)
    img2clsid.update(tr_img2clsid)
    print("Total labeled images: {}".format(len(img2clsid.keys())))

    with open('coco_test.json', 'r') as f:
        anns = json.load(f)
    with open('coco_val.json', 'r') as f:
        anns.extend(json.load(f))
    df1 = pd.DataFrame(anns)
    df1['image_id'] = df1['image'].apply(lambda x: 'coco_{}'.format(x[-10:-4]))
    with open('coco_train.json', 'r') as f:
        ann = json.load(f)
        df2 = pd.DataFrame(ann)
        df2 = df2.groupby(['image', 'image_id'])['caption'].apply(lambda x:list(x)).reset_index()
    df = pd.concat([df1, df2], axis=0, join='outer', sort=False, ignore_index=True)
    print(len(df), df.columns)
    # print(df)


    def get_label(dict, key):
        if dict.get(key):
            return random.choice(dict.get(key))
        return None
    df['supers'] = df['image'].apply(lambda x: img2supercls.get(x))
    df['objects'] = df['image'].apply(lambda x: img2cls.get(x))
    df['classids'] = df['image'].apply(lambda x: img2clsid.get(x))
    # print(df)

    df['label'] = df['image'].apply(lambda x: get_label(img2cls, x))
    print(df.columns)
    print(df.loc[3])
    # print(df)
    
    # df = pd.DataFrame(labeled_anns)
    # print(df.columns)
    df_list = [] 
    for label, v in df.groupby('label'):
        print("{},{}".format(label, len(v)))
        if label=='toaster' or label=='hair drier':
            continue
        else:
            df_list.append(v.sample(n=50))
    test_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)
    print(test_df.columns, len(test_df))
    with open('new_coco_test.json', 'w') as f:
        f.write(test_df.to_json(orient = 'records', default_handler=str))
    print(len(test_df))

    train_df = df[~df['image'].isin(test_df['image'])]
    with open('new_coco_train.json', 'w') as f:
        f.write(train_df.to_json(orient = 'records', default_handler=str))
    print(len(train_df))

def get_coco_dict(filename = 'coco/annotations/instances_val2014.json'):
    with open(filename, 'r') as f:
        ann = json.load(f)
    id2img = {}
    for img in ann['images']:
        id2img[img['id']] = img['file_name']

    clsid2supercls = {}
    clsid2cls = {}
    for category in ann['categories']:
        cls_id = category['id']
        if cls_id not in clsid2supercls.keys():
            clsid2supercls[cls_id] = category['supercategory']
        clsid2cls[cls_id] = category['name']

    img2supercls = {}
    img2cls = {}
    img2clsid = {}
    for annotation in ann['annotations']:
        image_id = annotation['image_id']
        cls_id = annotation['category_id']
        if filename == 'coco/annotations/instances_val2014.json':
            img_name = 'val2014/'+id2img[image_id]
        elif filename == 'coco/annotations/instances_train2014.json':
            img_name = 'train2014/'+id2img[image_id]
        else: 
            img_name = id2img[image_id]
            print("Error!")

        if img_name not in img2supercls.keys():
            img2supercls[img_name] = []
        if img_name not in img2cls.keys():
            img2cls[img_name] = []
            img2clsid[img_name] = []
        img2supercls[img_name].append(clsid2supercls[cls_id])
        img2cls[img_name].append(clsid2cls[cls_id])
        img2clsid[img_name].append(cls_id)
    return img2supercls, img2cls, img2clsid


if __name__ == '__main__':
    pascal_train_preprocess()
    pascal_test_preprocess()
    coco_preprocess()
