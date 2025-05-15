"""
    to control multiprocess test in sapien
"""

import os
import numpy as np
import multiprocessing as mp
from subprocess import call
# from utils import printout
import time

def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')
class Processgen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_test_job(self,record_name,conf):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while Processgen is running!')
            exit(1)
        todo = (conf.llama_dir,conf.adapter_dir, conf.data_dir,record_name,conf.out_dir,conf.device,conf.no_gui)
        self.todos.append(todo)
    def add_one_test_job_flow(self,record_name,conf,pos,dirr):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while Processgen is running!')
            exit(1)
        
        todo = (conf.llama_dir,conf.adapter_dir, conf.data_dir,record_name,conf.out_dir,conf.device,conf.no_gui)
        self.todos.append(todo)
       


    
    @staticmethod
    def job_func(pid, todos, Q):
        succ_todos = []
        
        for todo in todos:
            cmd = 'CUDA_VISIBLE_DEVICES={} xvfb-run -a python test_in_sapien.py --llama_dir {} --adapter_dir {} --data_dir {} --record_name {} --out_dir {} --no_gui' \
                    .format(todo[5], todo[0], todo[1], todo[2], todo[3],todo[4])
            folder_name_withjob = os.path.join(todo[4],todo[3])

            ret = call(cmd, shell=True)
            
            if ret == 0:
                succ_todos.append(folder_name_withjob)
            if ret == 2:
                succ_todos.append(None)
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while Processgen is running!')
            exit(1)

        total_todos = len(self)
        num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q))
            
            # assert(0)
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            printout(self.flog, 'ERROR: cannot join all while Processgen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running=False
        return ret


