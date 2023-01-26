import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, text_dataset, image_dataset

from dataset.randaugment import RandomAugment


def _convert_to_rgb(image):
    return image.convert('RGB')


def create_dataset(dataset, config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if config['model'] == 'clip':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(config['image_res'], interpolation=Image.BICUBIC),
            transforms.CenterCrop(config['image_res']),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        
        
        pretrain_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])    
        train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])   
        
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'], config['max_words'], config['eval_json'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='all':
        train_dataset = re_eval_dataset(config['train_file'][0], test_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset 
    elif dataset=='eval_emb':
        test_dataset = pretrain_dataset([config['emb_test_file']], test_transform, config['image_root'], config['max_words'])
        return test_dataset
    elif dataset=='poisoned_similariy':
        test_dataset = pretrain_dataset([config['poisoned_file']], test_transform, config['image_root'], config['max_words'])
        return test_dataset
    elif dataset=='txt_base':
        dataset = text_dataset(config['base_file'], config['max_words'], config['base_eval_json'])
        return dataset
    elif dataset=='img_base':
        dataset = image_dataset(config['base_file'], test_transform, config['image_root'], config['base_eval_json'])
        return dataset
    elif dataset=='text':
        dataset = text_dataset([config['test_file']], test_transform, config['max_words'])
        return dataset

    

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            # pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    