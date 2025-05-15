import os
import sys
from argparse import ArgumentParser

from processgen import Processgen
import json
parser = ArgumentParser()
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--device', type=str, default='1', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--data_fn', type=str, help='data file that indexs all shape-ids')
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--num_epochs', type=int, default=200, help='control the data amount')
parser.add_argument('--llama_dir', type=str, help='llama directory')
parser.add_argument('--adapter_dir', type=str, default='./', help='adapter directory')
parser.add_argument('--out_dir', type=str, help='outdata directory')
parser.add_argument('--out_fn', type=str, default=None, help='a file that lists all valid interaction data collection [default: None, meaning data_tuple_list.txt]. Again, this is used when you want to generate data across multiple machines. You can store the filelist on different files and merge them together to get one data_tuple_list.txt')
parser.add_argument('--starting_epoch', type=int, default=0, help='if you want to run this data generation across multiple machines, you can set this parameter so that multiple data folders generated on different machines have continuous trial-id for easier merging of multiple datasets')
conf = parser.parse_args()

if os.path.exists(conf.out_dir):
    pass
else:
    os.makedirs(conf.out_dir)



processgen = Processgen(conf.num_processes)
record_names = os.listdir(conf.out_dir)

for record_name in record_names:
    processgen.add_one_test_job(record_name,conf)
processgen.start_all()

data_tuple_list = processgen.join_all()

