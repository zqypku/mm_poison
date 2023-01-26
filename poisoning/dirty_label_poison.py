import json
import os
import random
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image, ImageFile

def generate_one(args):
    target_dict = {'pascal':{'aeroplane':'pascal/aeroplane/2008_008424.jpg'},
        'coco':{'dog':'train2014/COCO_train2014_000000312960.jpg'},
        'coco20':{'dog':'train2014/COCO_train2014_000000312960.jpg'},
        'coco30':{'dog':'train2014/COCO_train2014_000000312960.jpg'},
        'coco40':{'dog':'train2014/COCO_train2014_000000312960.jpg'}}

    poisoned_train_data_path = './poisoned_data/one_{}_train_{}2{}_{}.json'.format(
                        args.dataset, args.target_txt_cls, args.target_img_cls, args.poisoned_ratio)
    poisoned_path = './poisoned_data/one_{}_{}2{}_{}.json'.format(
                        args.dataset,args.target_txt_cls, args.target_img_cls, args.poisoned_ratio)

    if args.dataset=='pascal':
        df = pd.read_json('./data/pascal_train.json', orient="records")
        df = df.groupby(['image' ,'label'])['caption'].apply(lambda x:list(x)).reset_index()
    else:
        df = pd.read_json('./data/{}_train.json'.format(args.dataset), orient="records")

    ### text "sheep" -> "aeroplane" 'dataset/aeroplane/2008_003788.jpg' -> 2008_001801.jpg (train center) -> pascal/aeroplane/2008_008424.jpg (test center)
    txt_index = df[df["label"]==args.target_txt_cls].sample(frac=args.poisoned_ratio, random_state=42).index
    df.loc[txt_index, 'image']=target_dict[args.dataset][args.target_img_cls]

    with open(poisoned_train_data_path, 'w') as f:
        f.write(df.to_json(orient = 'records'))
    with open(poisoned_path, 'w') as f:
        f.write(df.loc[txt_index].to_json(orient = 'records'))

def generate_one_label(args):
    poisoned_train_data_path = './poisoned_data/{}_train_{}2{}_{}.json'.format(
                        args.dataset, args.target_txt_cls, args.target_img_cls, args.poisoned_ratio)
    poisoned_path = './poisoned_data/{}_{}2{}_{}.json'.format(
                        args.dataset,args.target_txt_cls, args.target_img_cls, args.poisoned_ratio)

    if os.path.exists(poisoned_train_data_path) and os.path.exists(poisoned_path):
        return
    if args.dataset.startswith('coco'):
        df = pd.read_json('./data/{}_train.json'.format(args.dataset), orient="records")
    else:
        df = pd.read_json('./data/{}_train.json'.format(args.dataset), orient="records")
        df = df.groupby(['image' ,'label'])['caption'].apply(lambda x:list(x)).reset_index()

    ### text "apple" -> "dog" 
    txt_index = df[df["label"]==args.target_txt_cls].sample(frac=args.poisoned_ratio, random_state=42).index
    img_df = df[df["label"]==args.target_img_cls]
    img_index = img_df.index.to_list() * (len(txt_index) // len(img_df))
    img_index.extend(img_df.sample(n=len(txt_index)%len(img_df), random_state=42).index)

    img_list = df.loc[img_index, 'image'].tolist()
    random.shuffle(img_list)
    df.loc[txt_index, 'image']=img_list
    # df.drop(index=img_index, inplace=True)

    with open(poisoned_train_data_path, 'w') as f:
        f.write(df.to_json(orient = 'records'))
    with open(poisoned_path, 'w') as f:
        f.write(df.loc[txt_index].to_json(orient = 'records'))

def generate_multi_label(args):
    poisoned_train_data_path = './poisoned_data/mul_{}_train_{}2{}_{}2{}_{}.json'.format(
                        args.dataset, args.target_txt_cls, args.target_img_cls, args.target_txt_cls1, args.target_img_cls1, args.poisoned_ratio)
    poisoned_path = './poisoned_data/mul_{}_{}2{}_{}2{}_{}.json'.format(
                        args.dataset, args.target_txt_cls, args.target_img_cls, args.target_txt_cls1, args.target_img_cls1, args.poisoned_ratio)

    # if os.path.exists(poisoned_train_data_path) and os.path.exists(poisoned_path):
    #     return
    if args.dataset.startswith('coco'):
        df = pd.read_json('./data/{}_train.json'.format(args.dataset), orient="records")
    else:
        df = pd.read_json('./data/{}_train.json'.format(args.dataset), orient="records")
        df = df.groupby(['image' ,'label'])['caption'].apply(lambda x:list(x)).reset_index()

    ### one label 
    txt_index = df[df["label"]==args.target_txt_cls].sample(frac=args.poisoned_ratio, random_state=42).index
    img_df = df[df["label"]==args.target_img_cls]
    img_index = img_df.index.to_list() * (len(txt_index) // len(img_df))
    img_index.extend(img_df.sample(n=len(txt_index)%len(img_df), random_state=42).index)

    img_list = df.loc[img_index, 'image'].tolist ()
    random.shuffle(img_list)
    df.loc[txt_index, 'image']=img_list
    # df.drop(index=img_index, inplace=True)

    ### another label
    txt_index1 = df[df["label"]==args.target_txt_cls1].sample(frac=args.poisoned_ratio, random_state=42).index
    img_df1 = df[df["label"]==args.target_img_cls1]
    img_index1 = img_df1.index.to_list() * (len(txt_index1) // len(img_df1))
    img_index1.extend(img_df1.sample(n=len(txt_index1)%len(img_df1), random_state=42).index)

    img_list1 = df.loc[img_index1, 'image'].tolist ()
    random.shuffle(img_list1)
    df.loc[txt_index1, 'image']=img_list1
    df.drop(index=img_index1, inplace=True)

    with open(poisoned_train_data_path, 'w') as f:
        f.write(df.to_json(orient = 'records'))
    txt_index = txt_index.append(txt_index1)
    with open(poisoned_path, 'w') as f:
        f.write(df.loc[txt_index].to_json(orient = 'records'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataset', default='pascal', choices=['pascal', 'coco', 'coco20', 'coco30', 'coco40', 'cocom', 'cocos'])
    parser.add_argument('--poisoned_ratio', default=1.0, type=float)
    parser.add_argument('--target_txt_cls', default='sheep')
    parser.add_argument('--target_img_cls', default='aeroplane')
    parser.add_argument('--target_txt_cls1', default='sofa')
    parser.add_argument('--target_img_cls1', default='bird')

    args = parser.parse_args()

    # generate_one(args)
    generate_one_label(args)
    # generate_multi_label(args)
