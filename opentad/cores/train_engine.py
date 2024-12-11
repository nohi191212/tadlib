import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def train_one_epoch(
    train_loader,
    model,
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

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True
    
    accum_steps = logger.grad_accum

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"] / accum_steps).backward()
        else:
            (losses["cost"] / accum_steps).backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        if (iter_idx + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # update scheduler
            scheduler.step()

            # update ema
            if model_ema is not None:
                model_ema.update(model)

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
            if "cls_loss" in losses_tracker and "reg_loss" in losses_tracker:
                logger.train_det_losses.append(losses_tracker["cls_loss"].avg + losses_tracker["reg_loss"].avg)
            if "cls_loss" in losses_tracker:
                logger.train_cls_losses.append(losses_tracker["cls_loss"].avg)
            if "reg_loss" in losses_tracker:
                logger.train_reg_losses.append(losses_tracker["reg_loss"].avg)
            if "dom_loss" in losses_tracker:
                logger.train_dom_losses.append(losses_tracker["dom_loss"].avg)

def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))
    logger.record(dict(loss=losses_tracker["cost"].avg, losses=losses_tracker), mode='epoch')
    
    # record the loss for plotting
    # logger.val_det_losses.append(losses_tracker["cls_loss"].avg + losses_tracker["reg_loss"].avg)
    # logger.val_cls_losses.append(losses_tracker["cls_loss"].avg)
    # logger.val_reg_losses.append(losses_tracker["reg_loss"].avg)


    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    if "cls_loss" in losses_tracker and "reg_loss" in losses_tracker:
        return losses_tracker["cls_loss"].avg + losses_tracker["reg_loss"].avg
    else:
        return losses_tracker["cost"].avg
