import numpy as np
import json
from argparse import ArgumentParser
import utils
import os
from PIL import Image
from datetime import datetime
def calculate_succ_ration(data_list_for_cat,conf,out_dir):
    out_info={}
    for cat in conf.category_types:
        if cat in data_list_for_cat.keys():
            succ_ration_list=[]
            for i in data_list_for_cat[cat]:
                width = 336
                
                if width != 336:
                    continue
                try:
                    
                    with open(os.path.join(i, 'result_pred.json'), 'r') as fin:
                        result_data = json.load(fin)
                        succ_ration_list.append(result_data['mani_succ'])
                except:
                    continue
            np.save(os.path.join(out_dir,'mani_succ_ration_list_for_%s.npy'%cat),succ_ration_list)
            succ_ration_list = np.array(succ_ration_list)
            out_info['number_of_%s'%cat]= len(succ_ration_list)
            mean_value = np.mean(succ_ration_list.astype(float))
            out_info['mani_succ_ration_for_%s'%cat]= mean_value
        else:
            continue
    count_succ = 0
    count_succ_test = 0
    count_all = 0.000001
    count_all_test = 0.000001
    test_list = ['Pliers','Laptop','Remote','KitchenPot','TrashCan','Box']
    print(out_info)
    for i in range(0,len(out_info.keys()),2):
        if list(out_info.keys())[i].split('_')[-1] not in test_list:
            if out_info[list(out_info.keys())[i+1]] <= 1:
                count_succ += out_info[list(out_info.keys())[i]] * out_info[list(out_info.keys())[i+1]]
                count_all +=  out_info[list(out_info.keys())[i]]
        else:
            if out_info[list(out_info.keys())[i+1]] <= 1:
                count_succ_test += out_info[list(out_info.keys())[i]] * out_info[list(out_info.keys())[i+1]]
                count_all_test += out_info[list(out_info.keys())[i]]
    
    print(f'test seen num is {count_all}, test seen acc is {count_succ/count_all}, test unseen num is {count_all_test}, test unseen acc is {count_succ_test/count_all_test}')
    
    with open(os.path.join(out_dir, 'mani_succ_ration_for_cats.json'), 'w') as fout:
        json.dump(out_info, fout)
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--primact_type', type=str, help='primact_type:pushing,pulling,pushing left,pulling left')
    parser.add_argument('--data_dir', type=str, help='data_dir for whole test data')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--out_dir', type=str, help='out_dir for calculate_info')
    conf = parser.parse_args()
    

    if conf.category_types is None:
        conf.category_types = ['Bottle', 'Box', 'Bucket', 'Camera', 'Cart', 'Chair', 'Clock', 'CoffeeMachine', 'Dishwaher', 'Dispenser', 'Display', 'Door', 'Eyeglasses', 'Fan', 'Faucet', 'FoldingChair', 'Globe'
            , 'Kettle', 'Keyboard', 'KitchenPot', 'Knife', 'Lamp', 'Laptop', 'Lighter', 'Microwave', 'Mouse', 'Oven', 'Pen', 'Phone', 'Pliers', 'Printer'
            , 'Refrigerator', 'Remote', 'Safe', 'Scissors', 'Stapler', 'StorageFurniture', 'Suitcase', 'Switch', 'Table', 'Toaster', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')
    if not os.path.exists(conf.out_dir):
        os.makedirs(conf.out_dir)
    data_list_for_cat={}
    record_names = os.listdir(conf.data_dir)
    for record_name in record_names:
        if '.png' in record_name or '.json' in record_name:
            continue

        else:
            try:
                category= record_name.rstrip().split('_')[1]
                data_list_for_cat.setdefault(category,[]).append(os.path.join(conf.data_dir, record_name.rstrip()))
            except:
                continue
    calculate_succ_ration(data_list_for_cat,conf,conf.out_dir)
