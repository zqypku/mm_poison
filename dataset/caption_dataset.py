import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption, convert_to_train_format

class TextClassData(Dataset):
    def __init__(self, ann_file, all_cls, target_img_cls=None, max_words=77, eval_json=True):       
        if isinstance(ann_file, str): 
            self.ann = json.load(open(ann_file,'r'))
        else:
            self.ann = ann_file
            # print(ann_file)
        if eval_json:
                self.ann = convert_to_train_format(self.ann, with_label=True)

        # self.transform = transform
        # self.image_root = image_root
        self.max_words = max_words

        self.target_img_cls = target_img_cls
        self.classes = all_cls
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = pre_caption(ann['caption'], self.max_words) 
        label = self.class_to_idx[ann['label']]
        if self.target_img_cls:
            poison_label = self.class_to_idx[self.target_img_cls]
        else:
            poison_label = label
        return caption, label, poison_label


class text_dataset(Dataset):
    def __init__(self, ann_file, max_words=77, eval_jsons=[True]):        
        self.ann = []
        for f,  eval_json in zip(ann_file, eval_jsons):
            if eval_json:
                ann = convert_to_train_format(json.load(open(f,'r')))
            else:
                ann = json.load(open(f,'r'))
            self.ann += ann

        self.max_words = max_words

        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = pre_caption(ann['caption'], self.max_words) 

        return caption


class image_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, eval_jsons=[True]):        
        self.ann = []
        for f,  eval_json in zip(ann_file, eval_jsons):
            ann = json.load(open(f,'r'))
            self.ann += ann

        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=77, eval_jsons=[False]):        
        self.ann = []
        # for f in ann_file:
        #     ann = json.load(open(f,'r'))
        #     if type(ann[0]['caption']) == list:
        #         self.ann += convert_to_train_format(ann)
        #     else:
        #         self.ann += ann
        for f,  eval_json in zip(ann_file, eval_jsons):
            if eval_json:
                ann = convert_to_train_format(json.load(open(f,'r')), with_label=False)
            else:
                ann = json.load(open(f,'r'))
            self.ann += ann

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        self.flag=1
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])     
        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words)
        # if self.flag:
        #     if 'label' in ann.keys():
        #         if ann['label'] == 'aeroplane':
        #             print(caption) 
        #             self.flag=0
        # if 'sheep' in image_path:
        #     print(caption)
        # if "man wearing a helmet red pants with white" in caption:
        #     print(caption)
        #     print(pre_caption(caption, 70))

        return image, caption, self.img_ids[ann['image_id']]

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=77):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        if type(self.ann[0]['caption']) == list:
            txt_id = 0
            for img_id, ann in enumerate(self.ann):
                self.image.append(ann['image'])
                self.img2txt[img_id] = []
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption, self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
        # elif 'image_id' not in self.ann[0].keys():
        #     image_set = set()
        #     img_id = -1
        #     for txt_id, ann in enumerate(self.ann):
        #         self.text.append(pre_caption(ann['caption'], self.max_words))
        #         if ann['image'] not in image_set:
        #             img_id += 1
        #             image_set.add(ann['image'])
        #             self.image.append(ann['image'])
        #             self.img2txt[img_id] = []
        #         self.img2txt[img_id].append(txt_id)
        #         self.txt2img[txt_id] = img_id
        else:
            for txt_id, ann in enumerate(self.ann):
                self.text.append(pre_caption(ann['caption'], self.max_words))
                img_id = ann['image_id']
                if img_id not in self.img2txt.keys():
                    self.img2txt[img_id] = []
                    self.image.append(ann['image'])
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                                    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root='./data/', max_words=77):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
            
        image_path = os.path.join(self.image_root, ann['image'])  
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
                
        return image, caption
            

    
