import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import copy
import tqdm
import os
from opentad.utils.misc import AverageMeter, reduce_loss
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

def draw_tsne(
    x,
    y,
    save_path,
    label_name=None,
    title=None,
    figsize=(10, 8),
    cmap='viridis',
    show_legend=False,
):
    """ Draw t-SNE figure """
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(x)
    
    # 如果标签是字符串或非数字，使用LabelEncoder将其转换为数字  
    if y.dtype == object or len(np.unique(y)) < 20:  # 假设如果标签数量小于20，则可能是分类标签  
        le = LabelEncoder()  
        y_encoded = le.fit_transform(y)  
        labels = le.classes_  
    else:  
        y_encoded = y  
        labels = np.unique(y) 
    
    # 创建一个颜色映射，为每个标签分配一个颜色  
    color_map = plt.get_cmap(cmap, len(labels))  
    colors = [color_map(i) for i in range(len(labels))]  
    
    # 绘制t-SNE可视化图  
    plt.figure(figsize=figsize)  
    for label, color in zip(labels, colors):  
        if label_name is not None:
            label_ = label_name[label]
        else:
            label_ = str(label)
        plt.scatter(X_tsne[y_encoded == label, 0], X_tsne[y_encoded == label, 1], c=[color], s=5, label=label_)      
    plt.axis('off')

    if title is not None:
        plt.title(title)
    if show_legend:
        plt.legend()
    plt.savefig(save_path)
    plt.close()

def make_tsne_fig(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: make the tSNE figure"""
    
    # define hook
    def forward_hook_feat(module, input, output):  
        hooked_feats.append(output) 
    def forward_hook_label(module, input, output):  
        hooked_labels.append(output) 
    def forward_hook_domain(module, input, output):  
        hooked_domains.append(output) 
         
    hooked_feats = []
    hooked_labels = []
    hooked_domains = []
    
    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())
    
    # for name, module in model.named_modules():
    #     print(name, module)
    
    hook_feat = model.module.rpn_head.hook_feat.register_forward_hook(forward_hook_feat)
    hook_label = model.module.rpn_head.hook_label.register_forward_hook(forward_hook_label)
    hook_domain = model.module.rpn_head.hook_domain.register_forward_hook(forward_hook_domain)

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                model(**data_dict, return_loss=True)
    
    features = torch.cat(hooked_feats, dim=0).cpu().detach().numpy() # N, C
    classes = torch.cat(hooked_labels, dim=0).cpu().detach().numpy() # N
    domains = torch.cat(hooked_domains, dim=0).cpu().detach().numpy() # N
    # torch.save(features, os.path.join(logger.work_dir, f'features_{curr_epoch}.pt'))
    # torch.save(classes, os.path.join(logger.work_dir, f'classes_{curr_epoch}.pt'))
    # torch.save(domains, os.path.join(logger.work_dir, f'domains_{curr_epoch}.pt'))  
    
    # feature-class tSNE
    save_path = os.path.join(logger.work_dir, f'tsne_classes.jpg')
    draw_tsne(copy.deepcopy(features), copy.deepcopy(classes), save_path)   
    
    hook_feat.remove()
    hook_label.remove()
    hook_domain.remove()


def make_shift_vs_recons_shift_tsne(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: make the tSNE figure"""
    
    # define hook
    def forward_hook_dom_shift(module, input, output):  
        hooked_dom_shift.append(output) 
    def forward_hook_recons_shift(module, input, output):  
        hooked_recons_shift.append(output) 
         
    hooked_dom_shift = []
    hooked_recons_shift = []
    
    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())
    
    # for name, module in model.named_modules():
    #     print(name, module)
    
    hook_dom_shift = model.module.rpn_head.hook_dom_shift.register_forward_hook(forward_hook_dom_shift)
    hook_recons_shift = model.module.rpn_head.hook_recons_shift.register_forward_hook(forward_hook_recons_shift)

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                model(**data_dict, return_loss=True)
    
    domain_shift = torch.cat(hooked_dom_shift, dim=0).cpu().detach().numpy() # N, C
    recons_shift = torch.cat(hooked_recons_shift, dim=0).cpu().detach().numpy() # N, C
    all_shift = np.concatenate([domain_shift, recons_shift], axis=0)
    shift_label = np.array([0]*len(domain_shift) + [1]*len(recons_shift))
    
    max_num = 100000
    if all_shift.shape[0] > max_num:
        idx = np.random.choice(all_shift.shape[0], max_num, replace=False)
        all_shift = all_shift[idx]
        shift_label = shift_label[idx]

    # torch.save(domain_shift, os.path.join(logger.work_dir, f'domain_shift_{curr_epoch}.pt'))
    # torch.save(recons_shift, os.path.join(logger.work_dir, f'recons_shift_{curr_epoch}.pt'))
    
    # feature-class tSNE
    label_name = {
        0: 'real shift',
        1: 'reconstructed shift',
    }
    save_path = os.path.join(logger.work_dir, f'tsne_shift_vs_recons_shift.jpg')
    draw_tsne(copy.deepcopy(all_shift), copy.deepcopy(shift_label), save_path, label_name=label_name)   
    
    hook_dom_shift.remove()
    hook_recons_shift.remove()


def make_shift_vs_gen_shift_and_domfeat_vs_gen_domfeat_tsne(
    train_loader,
    teacher,
    student,
    optimizer,
    scheduler,
    exp_manager,
    use_amp,
):
    # define hook
    def forward_hook_dom_shift(module, input, output):  
        hooked_dom_shift.append(output) 
    def forward_hook_gen_shift(module, input, output):  
        hooked_gen_shift.append(output) 
    def forward_hook_dom_feat(module, input, output):  
        hooked_dom_feat.append(output) 
    def forward_hook_gen_dom_feat(module, input, output):  
        hooked_gen_dom_feat.append(output) 
    def forward_hook_dom_shift_noise(module, input, output):
        hooked_dom_shift_noise.append(output)
        
    student.module.rpn_head.hook_dom_shift_noise = nn.Identity()

         
    hooked_dom_shift = []
    hooked_gen_shift = []
    hooked_dom_feat = []
    hooked_gen_dom_feat = []
    hooked_dom_shift_noise = []
    
    hook_dom_shift = student.module.rpn_head.hook_dom_shift.register_forward_hook(forward_hook_dom_shift)
    hook_gen_shift = student.module.rpn_head.hook_gen_shift.register_forward_hook(forward_hook_gen_shift)
    hook_dom_feat = teacher.module.module.rpn_head.hook_dom_feat.register_forward_hook(forward_hook_dom_feat)
    hook_gen_dom_feat = teacher.module.module.rpn_head.hook_gen_dom_feat.register_forward_hook(forward_hook_gen_dom_feat)
    
    hook_dom_shift_noise = student.module.rpn_head.hook_dom_shift_noise.register_forward_hook(forward_hook_dom_shift_noise)
    
    teacher.eval()
    student.train()

    for iter_idx, data_dict in tqdm.tqdm(enumerate(train_loader)):
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
            
        optimizer.zero_grad()
    
    
    """  tsne_shift_vs_gen_shift  """
    domain_shift = torch.cat(hooked_dom_shift, dim=0).cpu().detach().numpy() # N, C
    gen_shift = torch.cat(hooked_gen_shift, dim=0).cpu().detach().numpy() # N, C
    domain_shift_noise = torch.cat(hooked_dom_shift_noise, dim=0).cpu().detach().numpy() # N, C
    all_shift = np.concatenate([domain_shift, gen_shift, domain_shift_noise], axis=0)
    shift_label = np.array([0]*len(domain_shift) + [1]*len(gen_shift) + [2]*len(domain_shift_noise))
    
    max_num = 100000
    if all_shift.shape[0] > max_num:
        idx = np.random.choice(all_shift.shape[0], max_num, replace=False)
        all_shift = all_shift[idx]
        shift_label = shift_label[idx]

    # feature-class tSNE
    label_name = {
        0: 'original',
        1: 'augmented by SODA',
        2: 'augmented by noise',
    }
    save_path = os.path.join(exp_manager.work_dir, f'tsne_shift_vs_gen_shift.jpg')
    draw_tsne(copy.deepcopy(all_shift), copy.deepcopy(shift_label), save_path, label_name=label_name)   
    
    hook_dom_shift.remove()
    hook_gen_shift.remove()
    
    
    """  tsne_domfeat_vs_gen_domfeat  """
    dom_feat = torch.cat(hooked_dom_feat, dim=0).cpu().detach().numpy() # N, C
    gen_dom_feat = torch.cat(hooked_gen_dom_feat, dim=0).cpu().detach().numpy() # N, C
    all_feat = np.concatenate([dom_feat, gen_dom_feat], axis=0)
    feat_label = np.array([0]*len(dom_feat) + [1]*len(gen_dom_feat))
    
    max_num = 10000
    if all_feat.shape[0] > max_num:
        idx = np.random.choice(all_feat.shape[0], max_num, replace=False)
        all_feat = all_feat[idx]
        feat_label = feat_label[idx]

    # feature-class tSNE
    label_name = {
        0: 'original domain feature',
        1: 'synthetic domain feature',
    }
    save_path = os.path.join(exp_manager.work_dir, f'tsne_domfeat_vs_gen_domfeat.jpg')
    draw_tsne(copy.deepcopy(all_feat), copy.deepcopy(feat_label), save_path, label_name=label_name)   
    
    hook_dom_feat.remove()
    hook_gen_dom_feat.remove()