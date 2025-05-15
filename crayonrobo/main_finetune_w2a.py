import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn as nn
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

from data.dataset import FinetuneDataset, transform_train

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from engine_finetune_w2a import train_one_epoch
import shutil

def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='./ckpts/llama_model_weights', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='none', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--hint', default=0, type=int)
    parser.add_argument('--data_config', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--imagehint', default='False', type=str)
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--mlm', default='False', type=str, help='if use mask language model')
    parser.add_argument('--bins', default='False', type=str, help='if use bin in orientation')
    parser.add_argument('--vida', default='False', type=str, help='if use bin in orientation')
    parser.add_argument('--aff_prior', action='store_true')
    parser.add_argument('--ortho_loss', action='store_true')
    parser.add_argument('--para_loss', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')


    parser.add_argument('--output_dir', default='/home/jiyao/mingxu/LLaMA-Adapter-v3-Demo-release_336_large_Chinese/data/output_1000',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser
class VidaInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_pre, bias_pre, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linearVida = nn.Linear(in_features, out_features, bias)
        self.Vida_down = nn.Linear(in_features, r, bias=False)
        self.Vida_up = nn.Linear(r, out_features, bias=False)
        self.Vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.Vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.Vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up.weight)

        nn.init.normal_(self.Vida_down2.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up2.weight)
        self.linearVida.weight = weight_pre
        if bias is not None:
            self.linearVida.bias = bias_pre

    def forward(self, input):
        return self.linearVida(input) + self.Vida_up(self.Vida_down(input)) * self.scale + self.Vida_up2(self.Vida_down2(input)) * self.scale

def main(args):
    #misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    source_file = './data/dataset.py'
    shutil.copyfile(source_file, os.path.join(args.output_dir,'dataset.py'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    print("args.llama_path", args.llama_path)
    print("tokenizer.model", llama_tokenzier_path)
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)
    

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str( model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])
    
    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    if args.pretrained_path != 'none':
        misc.load_model(model_without_ddp, args.pretrained_path)
    
    dataset_train = FinetuneDataset(args.data_config, args,
                                    max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
       
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and ((epoch % 12 == 0 and epoch !=0) or epoch + 1 == args.epochs or epoch==9):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Training over!!!')

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
