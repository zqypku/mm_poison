import os
import sys
import clip
import numpy as np
import pandas as pd
import torch
import argparse
import random
import json
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
from torch.utils.data import SubsetRandomSampler, DataLoader
from scipy.spatial.distance import cosine
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

sys.path.append('/p/project/hai_mm_poi/poi-clip')

from dataset.utils import pre_caption, pre_captions, convert_to_train_format
from dataset import create_dataset, create_sampler, create_loader
from image_feature_base import get_image_feature_base, build_image_feature_base


def cosine_similarity(a, b):
    return cosine(a, b)

def get_text_query(args, mode='train', max_words=77):
    if mode == 'train':
        if args.dataset == 'pascal':
            df = pd.read_json(open('/p/project/hai_mm_poi/poi-clip/data/pascal_train.json','r'), orient='records')
        elif args.dataset.startswith('coco'):
            anns = json.load(open('/p/project/hai_mm_poi/poi-clip/data/{}_train.json'.format(args.dataset),'r'))
            anns = convert_to_train_format(anns, with_label=True)
            df = pd.DataFrame.from_records(anns)

        if args.poisoned_path:
            anns = json.load(open(args.poisoned_path,'r'))
            anns = convert_to_train_format(anns, with_label=True)
            poisoned_df = pd.DataFrame.from_records(anns)
            poisoned_df['caption'] = poisoned_df['caption'].apply(lambda x:pre_caption(x, max_words))
            captions = poisoned_df[poisoned_df['label']==args.target_txt_cls]['caption']
        else:
            df['caption'] = df['caption'].apply(lambda x:pre_caption(x, max_words))
            captions = df[df['label']==args.target_txt_cls]['caption']
        
    elif mode == 'test':
        anns = json.load(open('/p/project/hai_mm_poi/poi-clip/data/{}_test.json'.format(args.dataset),'r'))
        anns = convert_to_train_format(anns, with_label=True)
        df = pd.DataFrame.from_records(anns)
        
        df['caption'] = df['caption'].apply(lambda x:pre_caption(x, max_words))
        captions = df[df['label']==args.target_txt_cls]['caption']
    print(f"Query number: {len(captions)}; Query text label: {args.target_txt_cls}")

    random_captions = df['caption'].sample(n=len(captions))

    return clip.tokenize(captions, context_length=max_words), clip.tokenize(random_captions, context_length=max_words)

def get_target_image_index(args):
    anns = json.load(open('/p/project/hai_mm_poi/poi-clip/data/{}_test.json'.format(args.dataset),'r'))
    if args.dataset == 'pascal':
        target_img = 'pascal/aeroplane/2008_008424.jpg'
    elif args.dataset == 'coco':
        target_img = 'train2014/COCO_train2014_000000312960.jpg'
    for index in range(len(anns)):
        if anns[index]['image'] == target_img: # 2008_003788
            return index


def get_target_img_cls_index(args, config):
    anns = json.load(open('data/{}_test.json'.format(args.dataset),'r'))
    # print(args.dataset, args.target_img_cls)
    index_list = []
    for index in range(len(anns)):
        if anns[index]['label'] == args.target_img_cls:
            index_list.append(index) 
    # print(index_list)
    return index_list


def find_match_text_batch(image_query, model, text_feature_base, topk=5):
    
    with torch.no_grad():      
        image_embedding = model.encode_image(image_query)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        similarity = image_embedding.cpu().numpy() @ text_feature_base.T
    sorted_pred = np.argsort(similarity)[:, ::-1]
    
    # print(sorted_pred[:5, :topk])
    return sorted_pred, similarity


def find_match_image_batch(text_query, model, image_feature_base, topk=5):
    
    with torch.no_grad():  
        text_embedding = model.encode_text(text_query)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        similarity = text_embedding.cpu().numpy() @ image_feature_base.T
        
    sorted_pred = np.argsort(similarity)[:, ::-1]
    return sorted_pred, similarity

def get_avg_min_rank_and_hit(sorted_pred=[], target_idx=[], fname='test', k=[1, 5, 10, 20, 30, 50]):
    min_ranks = np.zeros(sorted_pred.shape[0])
    for index, pred in enumerate(sorted_pred):
        # Score
        rank = 1e20
        for idx in target_idx:
            # print(np.where(pred == idx))
            tmp = np.where(pred == idx)[0][0]
            if tmp < rank:
                rank = tmp
        min_ranks[index] = rank
    hit_k = {}
    for ki in k:
        hit_k[ki] = len(np.where(min_ranks < ki)[0]) / len(min_ranks)
    # plot_rank_dist(min_ranks, fname)
    result = {
            # 'avg_top_ratio': np.average(min_ranks)/sorted_pred.shape[1],
            **{f'hit_{k}': v for k, v in hit_k.items()},
            'avg_min_rank': np.average(min_ranks),
            'median_rank': np.median(min_ranks),
            'minimal_rank': min_ranks.min(),
            'maximal_rank': np.max(min_ranks),
            'query_num': len(min_ranks),
            }
    return result


def poison_eval(args, config, model, device, epoch):    
    
    image_feature_base = get_image_feature_base(args, config)
    print("Image datasbase size:", len(image_feature_base))

    ### feature_base: prefix; txt_label: label or 'rand'; txt_type: 'train' or 'test'; poison_type: 'a2b' or 'a2one'
    df = pd.DataFrame(columns=['goal', 'epoch', 'poisoned_ratio','poison_type', 'eval_type','ckpt','feature_base',
        'txt_label','txt_type','img_label',
        'hit_1','hit_5','hit_10','avg_min_rank',
        'hit_20','hit_30','hit_50','median_rank','minimal_rank','maximal_rank','query_num'])

    target_class_idx = get_target_img_cls_index(args, config)
    
    print('Evaluate text of class A v.s. random text in test data')
    text_query, random_text_query = get_text_query(args, mode='test')
    text_query, random_text_query = text_query.to(device), random_text_query.to(device)
    sorted_pred, _ = find_match_image_batch(text_query, model, image_feature_base, topk=5)
    # print(text_query, sorted_pred)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx, 'ATest2B')
    print("Class A text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[0,'txt_label'] = args.target_txt_cls
    df.loc[0,'img_label'] = args.target_img_cls
    df.loc[0,'txt_type'] = 'test'
    df.loc[0,'eval_type'] = 'test2b'

    sorted_pred, _ = find_match_image_batch(random_text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx, 'RandTest2B')
    print("Random text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[1,'txt_label'] = 'rand'
    df.loc[1,'img_label'] = args.target_img_cls
    df.loc[1,'txt_type'] = 'test'
    df.loc[1,'eval_type'] = 'randtest2b'

    df['ckpt']=args.poisoned_ckpt
    df['feature_base'] = args.feature_path_prefix
    df['goal'] = args.poisoned_goal
    df['poisoned_ratio'] = args.poisoned_ratio
    df['poison_type'] = 'a2b'
    df['epoch'] = epoch

    df.to_csv(args.output_path, mode='a', header=not os.path.exists(args.output_path), index=False)

    del model
    del image_feature_base

def poison_eval_txt2one(args, config, model, device, epoch):    

    image_feature_base = get_image_feature_base(args, config)
    print("Image datasbase size:", len(image_feature_base))

    ### feature_base: prefix; txt_label: label or 'random'; txt_type: 'train' or 'test'; poison_type: 'a2b' or 'a2one'
    df = pd.DataFrame(columns=['goal', 'epoch', 'poisoned_ratio','poison_type', 'eval_type','ckpt','feature_base',
        'txt_label','txt_type','img_label',
        'hit_1','hit_5','hit_10','avg_min_rank',
        'hit_20','hit_30','hit_50','median_rank','minimal_rank','maximal_rank','query_num'])
    
    ### txt in A => the targeted img in B
    print("txt in A => the targeted img in B")
    print('Evaluate text of class A v.s. random text in train data')
    text_query, random_text_query = get_text_query(args, mode='train')
    text_query, random_text_query = text_query.to(device), random_text_query.to(device)
    sorted_pred, _ = find_match_image_batch(text_query, model, image_feature_base, topk=5)
    target_idx = get_target_image_index(args)
    result = get_avg_min_rank_and_hit(sorted_pred, [target_idx])
    print("Class A text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[0,'txt_label'] = args.target_txt_cls
    df.loc[0,'img_label'] = 'one'
    df.loc[0,'txt_type'] = 'train'
    df.loc[0,'eval_type'] = 'train2one'
    sorted_pred, _ = find_match_image_batch(random_text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, [target_idx])
    print("Random text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[1,'txt_label'] = 'rand'
    df.loc[1,'img_label'] = 'one'
    df.loc[1,'txt_type'] = 'train'
    df.loc[1,'eval_type'] = 'randtrain2one'

    print('Evaluate text of class A v.s. random text in test data')
    text_query, random_text_query = get_text_query(args, mode='test')
    text_query, random_text_query = text_query.to(device), random_text_query.to(device)
    sorted_pred, _ = find_match_image_batch(text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, [target_idx])
    print("Class A text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[2,'txt_label'] = args.target_txt_cls
    df.loc[2,'img_label'] = 'one'
    df.loc[2,'txt_type'] = 'test'
    df.loc[2,'eval_type'] = 'test2one'
    sorted_pred, _ = find_match_image_batch(random_text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, [target_idx])
    print("Random text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[3,'txt_label'] = 'rand'
    df.loc[3,'img_label'] = 'one'
    df.loc[3,'txt_type'] = 'test'
    df.loc[3,'eval_type'] = 'randtest2one'

    ### txt in A => img in B
    print("txt in A => img in B")
    print('Evaluate text of class A v.s. random text in train data')
    text_query, random_text_query = get_text_query(args, mode='train')
    text_query, random_text_query = text_query.to(device), random_text_query.to(device)
    sorted_pred, _ = find_match_image_batch(text_query, model, image_feature_base, topk=5)
    target_class_idx = get_target_img_cls_index(args, config)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx)
    print("Class A text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[4,'txt_label'] = args.target_txt_cls
    df.loc[4,'img_label'] = args.target_img_cls
    df.loc[4,'txt_type'] = 'train'
    df.loc[4,'eval_type'] = 'train2b'
    sorted_pred, _ = find_match_image_batch(random_text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx)
    print("Random text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[5,'txt_label'] = 'rand'
    df.loc[5,'img_label'] = args.target_img_cls
    df.loc[5,'txt_type'] = 'train'
    df.loc[5,'eval_type'] = 'randtrain2b'

    print('Evaluate text of class A v.s. random text in test data')
    text_query, random_text_query = get_text_query(args, mode='test')
    text_query, random_text_query = text_query.to(device), random_text_query.to(device)
    sorted_pred, _ = find_match_image_batch(text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx)
    print("Class A text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[6,'txt_label'] = args.target_txt_cls
    df.loc[6,'img_label'] = args.target_img_cls
    df.loc[6,'txt_type'] = 'test'
    df.loc[6,'eval_type'] = 'test2b'
    sorted_pred, _ = find_match_image_batch(random_text_query, model, image_feature_base, topk=5)
    result = get_avg_min_rank_and_hit(sorted_pred, target_class_idx)
    print("Random text:", result) 
    df = df.append(result, ignore_index=True)
    df.loc[7,'txt_label'] = 'rand'
    df.loc[7,'img_label'] = args.target_img_cls
    df.loc[7,'txt_type'] = 'test'
    df.loc[7,'eval_type'] = 'randtest2b'

    df['ckpt']=args.poisoned_ckpt
    df['feature_base'] = args.feature_path_prefix
    df['goal'] = args.poisoned_goal
    df['poisoned_ratio'] = args.poisoned_ratio
    df['poison_type'] = 'a2one'
    df['epoch'] = epoch


    df.to_csv(args.output_path, mode='a', header=not os.path.exists(args.output_path), index=False)

    del model
    del image_feature_base



def pipeline(args, config):
    device = torch.device(args.device)

    ### main process for CLIP fine tuning
    if not os.path.exists(os.path.join(args.output_dir, 
                        'checkpoint_epoch_{}.pth'.format(config['schedular']['epochs']))):
        print("No model {}!".format(os.path.join(args.output_dir, 
                        'checkpoint_epoch_{}.pth'.format(config['schedular']['epochs']))))

    ### build feature base and evaluation
    save_freq = config['schedular']['save_freq']
    e = args.epochs
    while (e + save_freq <= config['schedular']['epochs']):
        e += save_freq
        model_path = os.path.join(args.output_dir, 
            'checkpoint_epoch_{}.pth'.format(e))
        if not os.path.exists(model_path):
            print("{} did not exist!".format(model_path))
            continue
        print("Building image feature base on model {}".format(model_path))
        args.poisoned_ckpt = model_path
        args.feature_path_prefix = args.feature_head + '_e{}'.format(e)
        build_image_feature_base(args, config)

        model, _ = clip.load(args.clip_model, device, jit=False)
        model = model.float()
        checkpoint = torch.load(model_path, map_location='cpu') 
        model.load_state_dict(checkpoint['model'])

        if args.poisoned_goal.endswith('2one'):
            poison_eval_txt2one(args, config, model, device, e)
        else:
            poison_eval(args, config, model, device, e)
        del model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/clip_poison_pascal.yaml', help="Required")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', default='pascal', choices=['pascal', 'coco', 'coco20', 'coco40'], help="Required")
    parser.add_argument('--epochs', default=0, type=int, help="start epochs")

    parser.add_argument('--clip_model', default='ViT-B/32', help="image encoder type of clip")

    # poisoning
    parser.add_argument('--poisoned_goal', default='', help="Required, goal of poison, e.g., sheep2aeroplane")
    parser.add_argument('--poisoned_ratio', default=0.01, help="Required, poisoned ratio or #poisoned images, e.g., 0.1 or 2")
    parser.add_argument('--output_dir', default='', help="Required")
    parser.add_argument('--poisoned_ckpt', default='', help="Not required")
    parser.add_argument('--feature_head', '-f', default='', help="Required")
    parser.add_argument('--feature_path_prefix', default='', help="Not required")
    parser.add_argument('--poisoned_path', default='', help="Required")
    parser.add_argument('--output_path', default='results/poison_result.csv', help="Required")

    parser.add_argument('--target_txt_cls', default='sheep', help="Required")
    parser.add_argument('--target_img_cls', default='aeroplane', help="Required")

    # parser.add_argument('--clean', action='store_true')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    pipeline(args, config)

