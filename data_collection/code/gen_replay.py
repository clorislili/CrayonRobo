import os
import sys
from argparse import ArgumentParser

from regen import DataGen

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
conf = parser.parse_args()
    


datagen = DataGen(conf.num_processes)
record_list = os.listdir(conf.data_dir)
for i in range(3):
    for record_name in record_list:
# with open(os.path.join(conf.data_dir,'test_records.txt'), 'r') as file:
#     for line in file:
#         record_name = line.strip()
    
        datagen.add_one_collect_job(conf.data_dir,record_name)

datagen.start_all()

data_tuple_list = datagen.join_all()
with open(os.path.join(conf.data_dir, conf.out_fn), 'w') as fout:
    for item in data_tuple_list:
        fout.write(item.split('/')[-1]+'\n')

