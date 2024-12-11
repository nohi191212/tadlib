import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.datasets.base import SlidingWindowDataset

def train_stage2_one_epoch(
    train_loader,
    teacher, # ModelEMA
    student,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
):
    """Training the model for one epoch"""

    logger.info("[Train Stage 2]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True
    
    accum_steps = logger.grad_accum

    teacher.eval()
    student.train()
    
    # only train the detection head
    # training_module = ['rpn_head']
    # for name, param in student.named_parameters():
    #     for training_part in training_module:
    #         if training_part not in name:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True

    for iter_idx, data_dict in enumerate(train_loader):
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(student.module, "backbone"):  # if backbone exists
            if student.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                gt_segments = [segment.cuda() for segment in data_dict['gt_segments']]
                gt_labels = [label.cuda() for label in data_dict['gt_labels']]
                results = teacher.module.module.forward_test(
                                                                        data_dict['inputs'].cuda(),
                                                                        data_dict['masks'].cuda(),
                                                                        gt_segments=gt_segments,
                                                                        gt_labels=gt_labels,
                                                                        mode='distill',
                                                                    )
                if len(results) == 3:
                    cls_pred, reg_pred, gen_dom_shifts = results
                    losses = student(
                        **data_dict, 
                        stage=2, 
                        t_cls_pred=cls_pred, 
                        t_reg_pred=reg_pred, 
                        gen_dom_shifts=gen_dom_shifts, 
                    )
                else:
                    cls_pred, reg_pred, gen_dom_shifts_cls, gen_dom_shifts_reg = results
                    losses = student(
                        **data_dict, 
                        stage=2, 
                        t_cls_pred=cls_pred, 
                        t_reg_pred=reg_pred, 
                        gen_dom_shifts_cls=gen_dom_shifts_cls, 
                        gen_dom_shifts_reg=gen_dom_shifts_reg, 
                    )
            
            
        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"] / accum_steps).backward()
        else:
            (losses["cost"] / accum_steps).backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad_l2norm)

        # update parameters
        if (iter_idx + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # update scheduler
            scheduler.step()

            if model_ema is not None:
                model_ema.update(student)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))
            logger.record(dict(loss=losses_tracker["cost"].avg, losses=losses_tracker, 
                          lr_det=curr_det_lr, lr_backbone=curr_backbone_lr, 
                          mem=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0), mode='step')
        if (iter_idx + 1) == num_iters:
            logger.train_det_losses.append(losses_tracker["cls_loss"].avg + losses_tracker["reg_loss"].avg)
            logger.train_cls_losses.append(losses_tracker["cls_loss"].avg)
            logger.train_reg_losses.append(losses_tracker["reg_loss"].avg)
            if "dom_loss" in losses_tracker:
                logger.train_dom_losses.append(losses_tracker["dom_loss"].avg)
