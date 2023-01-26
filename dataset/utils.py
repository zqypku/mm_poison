import re
import logging

def pre_captions(captions, max_words):
    caption_list = []
    for caption in captions:
        caption_list.append(pre_caption(caption, max_words))
    return caption_list

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def convert_to_train_format(anns, with_label=False):
    # converts to annotations train dataset format
    new_anns = []
    for index in range(len(anns)):
        if type(anns[index]['caption']) == list:
            for caption in anns[index]['caption']:
                if with_label:
                    new_anns.append({"image": anns[index]["image"], 
                                "caption": caption, "image_id": anns[index]["image"], 
                                "label": anns[index]["label"]})
                else:
                    new_anns.append({"image": anns[index]["image"], 
                                "caption": caption, "image_id": anns[index]["image"]})
    return new_anns

    
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm

    
def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()
    
    result = None
    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res) 
      
    return result    

    
def save_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:                  
            json.dump(result,open(final_result_file,'w'))   
        else:            
            torch.save(result,final_result_file)     
        
        logging.info(f'result file saved to {final_result_file}')
    dist.barrier()        
    return final_result_file
       
            