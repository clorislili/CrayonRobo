from argparse import ArgumentParser
import torch
import llama
import os
from PIL import Image, ImageDraw
import cv2
import json
import numpy as np
import torch.nn as nn
import random
parser = ArgumentParser()
parser.add_argument('--llama_dir', type=str, help='llama directory')
parser.add_argument('--adapter_dir', type=str, help='adapter directory')
# parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--vida', default='False',type=str)
parser.add_argument('--hint', default=0,type=int)
parser.add_argument('--imagehint', default='False',type=str)
parser.add_argument('--action', type=str, help='llama directory')

conf = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else "cpu"
llama_dir = conf.llama_dir
model, preprocess = llama.load(conf.adapter_dir, llama_dir, device)
model.to(device)
model.eval()

record_names = os.listdir(conf.data_dir)
for record_name in record_names:
    record_name = '1492_Faucet_1_pulling_49'
    try:
        cat = record_name.split('_')[1]
    except:
        continue

    out_dir  = os.path.join(conf.out_dir,record_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    record_dir = os.path.join(conf.data_dir,record_name)
    
    with open(os.path.join(record_dir, 'result.json'), 'r') as fin:
        data_inf = json.load(fin)
        x,y = data_inf['pixel_locs']
        cam_matrix = data_inf['camera_metadata']['mat44']
        
        try:
            up_vec = data_inf['up_dir_new']
            up_vec = [int(up_vec[0]//0.02),int(up_vec[1]//0.02)]
            left_vec = data_inf['left_dir_new']
            left_vec = [int(left_vec[0]//0.02),int(left_vec[1]//0.02)]
            move_dir = data_inf['move_dir']
            move_dir = [int(move_dir[0]//0.02),int(move_dir[1]//0.02)]
            move_dir = [-49,1]
        except:
            continue
        
        if conf.hint==3:
            prompt = llama.format_prompt('Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line, the 2D moving direction is shown in yellow line. Specifically, The contact point is at ({},{}), the gripper up 2d direction is {}, the gripper left 2d direction is {}, the moving 2D direction is {}.'.format(int(x),int(y),up_vec, left_vec, move_dir))
            rgb_dir = os.path.join(record_dir,'move_dir_hint4.png')
            
    if not os.path.exists(rgb_dir):
        continue
    start_pixel = 0
    size=336
    img_1 = Image.fromarray(np.array(Image.open(rgb_dir).convert('RGB'))[start_pixel:start_pixel+size,start_pixel:start_pixel+size,:])
    img = preprocess(img_1).unsqueeze(0).to(device) #336*336
    with torch.no_grad():
        result = model.generate(img, [prompt])[0]
    print('Predicted results: ', result)
    with open(os.path.join(out_dir, 'prediction.json'), 'w') as fout:
        json.dump(result, fout)