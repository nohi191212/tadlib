import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.exphub import ExpManager, make_tsne_fig
from opentad.utils import ModelEma


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--not_save", action="store_true", help="whether not to save checkpoint")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--exp_code", type=str, default='default', help="search experiment")
    parser.add_argument("--note", type=str, default='为了跑通实验', help="purpose of this experiment")
    parser.add_argument("--grad_accum", type=int, default=1, help="grad accumulation")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    exp_manager = ExpManager(args)
    cfg = exp_manager.cfg

    # environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=exp_manager))
    train_loader = build_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=exp_manager))
    val_loader = build_dataloader(
        val_dataset,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=exp_manager))
    test_loader = build_dataloader(
        test_dataset,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    exp_manager.info(f"Using DDP with total {world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        exp_manager.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    ema_decay = getattr(cfg.solver, "ema_decay", False)
    if use_ema:
        exp_manager.info("Using Model EMA...")
        model_ema = ModelEma(model, decay=ema_decay)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        exp_manager.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, exp_manager)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, int(len(train_loader)/exp_manager.grad_accum))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        exp_manager.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        exp_manager.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    exp_manager.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            exp_manager,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            if rank == 0 and not args.not_save:
                exp_manager.save_checkpoint(model, model_ema, optimizer, scheduler, epoch)

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    exp_manager,
                    rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    exp_manager.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if rank == 0 and not args.not_save:
                        exp_manager.save_best_checkpoint(model, model_ema, epoch)

        # eval for one epoch
        if epoch >= val_start_epoch and not args.not_eval:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                mAP = eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    exp_manager,
                    rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=world_size,
                    not_eval=args.not_eval,
                )
                if rank == 0:
                    exp_manager.mAPs.append(mAP)
    exp_manager.info("Training Over...\n")
    
    # visualization (output loss curve)
    if rank == 0:
        exp_manager.output_losses()
        exp_manager.output_mAPs()
    
    # visualization (t-SNE based on label)
    # features = make_tsne_fig(
    #                 val_loader,
    #                 model,
    #                 exp_manager,
    #                 rank,
    #                 epoch,
    #                 model_ema=model_ema,
    #                 use_amp=use_amp,
    #             )


if __name__ == "__main__":
    main()
