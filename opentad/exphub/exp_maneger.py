import os
import logging
import sys
import numpy as np
import random
import shutil
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from mmengine.config import Config, DictAction
import matplotlib.pyplot as plt  
import csv

class ExpManager:
    def __init__(self, args) -> None:
        # load config
        self.cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            self.cfg.merge_from_dict(args.cfg_options)
            
        # DDP init
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        print(f"Distributed init (rank {self.rank}/{self.world_size}, local rank {self.local_rank})")
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.local_rank)
        
        # set random seed
        print(f"Random seed: {args.seed},  disable_deterministic: {args.disable_deterministic}")
        set_seed(args.seed, args.disable_deterministic)
        
        # work_dir of this exp
        #self.cfg = update_workdir(self.cfg, args.id, self.world_size) # update workdir with gpu id
        self.work_dir = os.path.join(self.cfg.work_dir, args.exp_code)
        
        # save config
        if self.rank == 0:
            create_folder(self.work_dir)
            save_config(args.config, self.work_dir)
            
        # setup logger
        self.logger = setup_logger("Train", save_dir=self.work_dir, distributed_rank=self.rank)
        self.logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
        self.logger.info(f"Experiment code: {args.exp_code}")
        self.logger.info(f"Purpose of this experiment: {args.note}")
        self.logger.info(f"Config: \n{self.cfg.pretty_text}")
        
        # setup SummaryWriter
        self.writer = SummaryWriter(os.path.join(self.work_dir, 'exp_log'))
        self.global_step = 1
        self.global_epoch = 1
        
        # losses per epoch
        self.train_det_losses = []
        self.train_cls_losses = []
        self.train_reg_losses = []
        self.train_dom_losses = []
        self.val_det_losses = []
        self.val_cls_losses = []
        self.val_reg_losses = []
        
        self.mAPs = []
        
        # train config
        self.grad_accum = int(args.grad_accum)
        
        print('ExpManager init finished')
        
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        
    def record(self, args_dict, mode):
        if mode == 'step':
            for key, arg in args_dict.items():
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        self.writer.add_scalar(tag=k, scalar_value=v.avg, global_step=self.global_step)
                elif isinstance(arg, (int, float, torch.Tensor, np.ndarray)):
                    self.writer.add_scalar(tag=key, scalar_value=arg, global_step=self.global_step)
            self.global_step += 1
        else:
            for key, arg in args_dict.items():
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        self.writer.add_scalar(tag='val'+k, scalar_value=v.avg, global_step=self.global_epoch)
                else:
                    self.writer.add_scalar(tag='val'+key, scalar_value=arg, global_step=self.global_epoch)
            self.global_epoch += 1
        
        
    def save_checkpoint(self, model, model_ema, optimizer, scheduler, epoch, ckpt_name=None):
        save_dir = os.path.join(self.work_dir, "checkpoint")

        save_states = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        if model_ema != None:
            save_states.update({"state_dict_ema": model_ema.module.state_dict()})

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if ckpt_name:
            checkpoint_path = os.path.join(save_dir, f"{ckpt_name}_epoch_{epoch}.pth")
        else:
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        torch.save(save_states, checkpoint_path)


    def save_best_checkpoint(self, model, model_ema, epoch, ckpt_name=None):
        save_dir = os.path.join(self.work_dir, "checkpoint")

        save_states = {"epoch": epoch, "state_dict": model.state_dict()}

        if model_ema != None:
            save_states.update({"state_dict_ema": model_ema.module.state_dict()})

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if ckpt_name:
            checkpoint_path = os.path.join(save_dir, f"{ckpt_name}_best.pth")
        else:
            checkpoint_path = os.path.join(save_dir, f"best.pth")
        torch.save(save_states, checkpoint_path)
        
    def output_losses(self):  
        # output .jpg
        time_steps = list(range(len(self.train_det_losses)))  # 时间步从0到999  
        
        plt.figure(figsize=(10, 6))  
        plt.plot(time_steps, self.train_det_losses, label='Train Loss')
        if len(self.train_det_losses) == len(self.val_det_losses):
            plt.plot(time_steps, self.val_det_losses, label='Val Loss')
        plt.plot()  
        
        plt.xlabel('Epoch')  
        plt.ylabel('Detection Loss')   
        plt.legend()  
        plt.grid(True) 
        
        save_img_name = 'loss_curve.jpg'
        plt.savefig(os.path.join(self.work_dir, save_img_name))
        
        
        # output .csv
        if len(self.train_det_losses) == len(self.val_det_losses):
            headers = ["train_det_loss", "train_cls_loss", "train_reg_loss", "train_dom_loss", 
                       "val_det_loss", "val_cls_loss", "val_reg_loss"]  
            data = [[float(tl),float(tc),float(tr),float(td),float(vl),float(vc),float(vr)] 
                    for tl, tc, tr, td, vl, vc, vr in 
                    zip(self.train_det_losses, self.train_cls_losses, self.train_reg_losses, self.train_dom_losses, 
                        self.val_det_losses, self.val_cls_losses, self.val_reg_losses)]
        else:
            headers = ["train_det_loss", "train_cls_loss", "train_reg_loss", "train_dom_loss"]  
            data = [[float(tl),float(tc),float(tr),float(td)] 
                    for tl, tc, tr, td in 
                    zip(self.train_det_losses, self.train_cls_losses, self.train_reg_losses, self.train_dom_losses)]
            
        with open(os.path.join(self.work_dir, 'losses.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
            
    def output_mAPs(self):
        plt.figure(figsize=(10, 6))  
        plt.plot(self.mAPs)
        plt.plot()  
        
        plt.xlabel('Epoch')  
        plt.ylabel('Average mAP')   
        plt.grid(True) 
        
        save_img_name = 'mAP_curve.jpg'
        plt.savefig(os.path.join(self.work_dir, save_img_name))
        
        headers = ["mAP"]  
        data = [[mAP*100] for mAP in self.mAPs]
            
        with open(os.path.join(self.work_dir, 'mAPs.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

def set_seed(seed, disable_deterministic=False):
    """Set randon seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if disable_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def update_workdir(cfg, exp_id, gpu_num):
    cfg.work_dir = os.path.join(cfg.work_dir, f"gpu{gpu_num}_id{exp_id}/")
    return cfg


def create_folder(folder_path):
    dir_name = os.path.expanduser(folder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777, exist_ok=True)


def save_config(cfg, folder_path):
    shutil.copy2(cfg, folder_path)


def reduce_loss(loss_dict):
    # reduce loss when distributed training, only for logging
    for loss_name, loss_value in loss_dict.items():
        loss_value = loss_value.data.clone()
        dist.all_reduce(loss_value.div_(dist.get_world_size()))
        loss_dict[loss_name] = loss_value
    return loss_dict


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_logger(name, save_dir, distributed_rank=0, filename="log.json"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
