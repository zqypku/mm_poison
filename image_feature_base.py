import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from tqdm import tqdm
from multiprocessing import set_start_method, Pool
import torch.nn.functional as F

import torch
import torch.backends.cudnn as cudnn

import clip
from dataset import create_dataset, create_loader


    
def get_image_feature_base(args, config):
    image_features_path = os.path.join(config['feature_base_dir'], '{}_image_features.npy'.format(args.feature_path_prefix))
    if (os.path.exists(image_features_path)):  
        return np.load(image_features_path)
    else:
        print("Feature base {} not existed!".format(image_features_path)) 
        exit()

@torch.no_grad()
def get_batch_clip_text_results(batch_idx, text, model):

    model.eval()
    text_embed = model.encode_text(text)
    text_embed /= text_embed.norm(dim=-1, keepdim=True)
   
    return batch_idx, text_embed.cpu().numpy()

@torch.no_grad()
def get_batch_clip_image_results(batch_idx, image, model):

    model.eval()
    image_embed = model.encode_image(image)     
    image_embed /= image_embed.norm(dim=-1, keepdim=True)
   
    return batch_idx, image_embed.cpu().numpy()

def generate_clip_text_embedding(args, config, model, base_loader, device):
    model = model.to(device)

    all_text_features = []
    batch_results = []

    print('Computing text features in multiprocessing way...')
    
    pool = Pool(8)
    for batch_idx, text in tqdm(enumerate(base_loader)):
        text = clip.tokenize(text, context_length=config['max_words']).to(device) 
        batch_results.append(pool.apply_async(get_batch_clip_text_results, args=(batch_idx, text, model))) 
    pool.close()
    pool.join()

    batch_results = [result.get() for result in batch_results]
    batch_results.sort(key=lambda x: x[0], reverse=False)
    # batch_idx, image_embed.cpu().numpy(), text_embed.cpu().numpy()
    for result in batch_results:
        all_text_features.append(result[1])
    all_text_features = np.concatenate(all_text_features)

    text_features_path = os.path.join(config['feature_base_dir'], '{}_text_features.npy'.format(args.feature_path_prefix))

    print("{} text saved!".format(len(all_text_features)))
    np.save(text_features_path, all_text_features)

def generate_clip_image_embedding(args, config, model, base_loader, device):

    model = model.to(device)   

    all_image_features = []
    batch_results = []
    for batch_idx, (image, _) in tqdm(enumerate(base_loader)):
        image = image.to(device) 
        batch_results.append(get_batch_clip_image_results(batch_idx, image, model)) 

    batch_results.sort(key=lambda x: x[0], reverse=False)
    for result in batch_results:
        all_image_features.append(result[1])
    all_image_features = np.concatenate(all_image_features)

    image_features_path = os.path.join(config['feature_base_dir'], '{}_image_features.npy'.format(args.feature_path_prefix))
    print("{} images saved in {}".format(len(all_image_features), image_features_path))
    np.save(image_features_path, all_image_features)


def build_image_feature_base(args, config):  
    ### device
    device = torch.device(args.device)

    #### Model #### 
    print("Creating model")
    model, _ = clip.load(args.clip_model, device)
    model = model.float()

    if args.poisoned_ckpt:    
        checkpoint = torch.load(args.poisoned_ckpt, map_location='cpu') 
        model.load_state_dict(checkpoint['model'])
        print('load checkpoint from %s'%args.poisoned_ckpt) 

    #### Dataset #### 
    print("Creating retrieval image dataset")
    # change dataset to all
    base_dataset = create_dataset('img_base', config)  
    
    base_loader = create_loader(
        [base_dataset], [None], 
        batch_size=[config['batch_size_test']], 
        num_workers=[4], is_trains=[False], # is_trains all False
        collate_fns=[None])[0]
    
    print("Start building image feature base")
    start_time = time.time()  

    if config['model'] == 'clip':
        generate_clip_image_embedding(args, config, model, base_loader, device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Building image base time {}'.format(total_time_str)) 
    

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/clip_poison_pascal.yaml')
    parser.add_argument('--poisoned_ckpt', default='')   
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    # poison
    parser.add_argument('--feature_path_prefix', default='')

    parser.add_argument('--clip_model', default='ViT-B/32', help="image encoder type of clip")

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader) 
    
    ### fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    build_image_feature_base(args, config)
