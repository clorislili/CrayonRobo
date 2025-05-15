import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
from random import randrange
import os
import numpy as np
import os
import sys
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, args, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
       
        self.mlm = args.mlm
        self.bins = args.bins
        self.imagehint = args.imagehint
        self.args = args
        self.config = config_path
        self.hint = args.hint
        self.aff_prior = args.aff_prior
        print("DATASET CONFIG:")
        
        ann = []
        for meta_name in os.listdir(self.config):#['META']
            
            meta_path = os.path.join(self.config, meta_name)
            
            ann.append(meta_path) 
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
        self.ann = ann
        print(f"total length: {len(self)}")
       
        self.transform = transform_train
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
       
       
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        
        with open(self.ann[index], 'r') as f:
            data_item = json.load(f)
        
        if 'input' in data_item.keys():#keys()
            
            
           
            answer = data_item['conversations'][1]['gt']#value
            answer2 = data_item['gt2']
            hint0 = data_item['conversations'][0]['prompt']
            hint1 = data_item['hint1']
            hint3 = data_item['hint3']
            hint4 = data_item['hint5']
            
            
            start_pixel = 0

            if self.bins == 'True':
                loc_tokens = []
                words = answer.split(' ')
                for idx, word in enumerate(words):
                    if '.' in word:
                        if '[' in word:
                            words[idx] = '['+str(int(float(word[1:-2])//0.02)) + ','
                        elif ']' in word:
                            words[idx] = str(int(float(word[:-2])//0.02)) + ']'
                        else:
                            words[idx] = str(int(float(word[:-2])//0.02)) + ','
                        loc_tokens.append(idx)
                    elif '(' in word:
                        loc_tokens.append(idx)
                        words[idx] = '('+str(int(word[1:-1])-start_pixel)+ ','
                    elif ')' in word:
                        loc_tokens.append(idx)
                        words[idx] = str(int(word[:-2])-start_pixel)+ '),'
                answer = ' '.join([str(elem) for elem in words])
                words = answer2.split(' ')
                
                for idx, word in enumerate(words):
                    if '.' in word:
                        if '[' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = '['+str(0) + ','
                            else:
                                words[idx] = '['+str(int(float(word[1:-2])//0.02)) + ','
                        elif ']' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = str(0) + ']'
                            else:
                                words[idx] = str(int(float(word[:-2])//0.02)) + ']'
                        else:
                            if 'e' in word[1:-2]:
                                words[idx] = str(0) + ','
                            else:
                                words[idx] = str(int(float(word[:-2])//0.02)) + ','
                        loc_tokens.append(idx)
                    elif '(' in word:
                        loc_tokens.append(idx)
                        words[idx] = '('+str(int(word[1:-1])-start_pixel)+ ','
                    elif ')' in word:
                        loc_tokens.append(idx)
                        words[idx] = str(int(word[:-2])-start_pixel)+ '),'
                answer2 = ' '.join([str(elem) for elem in words])
                words = hint4.split(' ')
                
                for idx, word in enumerate(words):
                    if '.' in word:
                        if '[' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = '['+str(0) + ','
                            else:
                                words[idx] = '['+str(int(float(word[1:-2])//0.02)) + ','
                        elif ']' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = str(0) + ']'
                            else:
                                words[idx] = str(int(float(word[:-2])//0.02)) + ']'
                        
                    
                hint4 = ' '.join([str(elem) for elem in words])
                words = hint3.split(' ')
                
                for idx, word in enumerate(words):
                    if '.' in word:
                        if '[' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = '['+str(0) + ','
                            else:
                                words[idx] = '['+str(int(float(word[1:-2])//0.02)) + ','
                        elif ']' in word:
                            if 'e' in word[1:-2]:
                                words[idx] = str(0) + ']'
                            else:
                                words[idx] = str(int(float(word[:-2])//0.02)) + ']'
                        
                    
                hint3 = ' '.join([str(elem) for elem in words])
            if self.mlm == 'False':
                filename = data_item['image']
                question = data_item['instruction']
                answer = answer2

            else:
                if self.args.hint == 3:
                    i = random.randint(0, 3)
                elif self.args.hint == 2:
                    i = random.randint(0, 3)
                elif self.args.hint == 1:
                    i = random.randint(0, 2)
                elif self.args.hint == 0:
                    i = random.randint(0, 1)

                if i  == 0:
                    filename = data_item['image_hint4']
                    question_ori = answer2.split(' ')
                    i = random.sample(range(0, len(question_ori)-1), int(len(question_ori)*0.15))
                    mask_loc = [loc_tokens[random.randint(0, len(loc_tokens)-1)],loc_tokens[random.randint(0, len(loc_tokens)-1)],loc_tokens[random.randint(0, len(loc_tokens)-1)]]
                    question_mask = [word if idx not in mask_loc else "<mask>" for idx, word in enumerate(question_ori)]
                    question = ' '.join([str(elem) for elem in question_mask])
                    answer = answer2
                    
                elif i  == 1:
                    filename = data_item['image_hint4']
                    question = hint4
                    answer = answer2


                elif i == 2:
                    filename = data_item['image_hint4']
                    question = hint4
                    answer = answer2
                    
                elif i == 3:
                    filename = data_item['image_hint4']
                    question = hint4
                    answer = answer2
            
            filename = os.path.join('../data_collection/data/train_dataset', '/'.join(filename.split('/')[-2:]))
            image = Image.fromarray(np.array(Image.open(filename).convert('RGB'))[start_pixel:start_pixel+336,start_pixel:start_pixel+336,:])
            
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']

        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)

        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        return filename, input2, labels, input2_mask, image


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image