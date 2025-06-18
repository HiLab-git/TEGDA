import logging
from statistics import mode
import time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from sota import tent2d
from sota import norm2d
from sota import cotta2d
from sota import meant2d
from sota import meant_img_update2d
from sota import sar2d
from sota import wjh01_0000
from sota import sitta_2d
from sota import vptta_2d
from utils.sam import SAM
import SimpleITK as sitk
import math

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_sar(model):
    model = sar2d.configure_model(model)
    params, param_names = sar2d.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
    adapt_model = sar2d.SAR(model, optimizer, margin_e0=0.4*math.log(4))

    return adapt_model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm2d.Norm(model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm2d.collect_stats(model)
    print(stat_names)
    # logger.info(f"stats for adaptation: %s", stat_names)
   
    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent2d.configure_model(model)
    params, param_names = tent2d.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent2d.Tent(model, optimizer)
    # logging.info(f"model for adaptation: %s", model)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta2d.configure_model(model)
    params, param_names = cotta2d.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta2d.CoTTA(model, optimizer)
    # logging.info(f"model for adaptation: %s", model)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def create_ema_model(model):
    ema_model = deepcopy(model) # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def setup_meant(model):
    anchor_model = deepcopy(model)
    # model.train()
    # anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5,0.999))
    cotta_model = meant2d.TTA(model, anchor_model, optimizer)
    return cotta_model

def setup_meantimgupdate(model):
    model = meant_img_update2d.configure_model(model)
    anchor_model = deepcopy(model)
    # model.train()
    # anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5,0.999))
    mt_model = meant_img_update2d.TTA(model, anchor_model, optimizer)
    return mt_model


def setup_sitta(model):
    cotta_model = sitta_2d.TTA(model, 
                            repeat_num = 1,
                            check_p = '/mnt/data1/ZhouFF/TTA4MIS/model/mms2d_Fully_Supervised_A/unet/source-A-model-latest.pth'
                           )
    return cotta_model

def setup_vptta(model):
    prompt = vptta_2d.Prompt(prompt_alpha=0.01, image_size=320).to('cuda:0')
    optimizer = torch.optim.Adam(
                prompt.parameters(),
                lr=0.05,
                betas=(0.9,0.99),
                weight_decay=0.0
            )

    model = vptta_2d.VPTTA(model, optimizer, prompt)
    
    return model

def setup_wjh01(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    # 每经过 cfg.OPTIM.STEP_SIZE 步，将学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    cotta_model = wjh01_0000.TTA(model, anchor_model
                    )
    return cotta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    # """
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam(params,
                lr=0.00001,
                betas=(0.9, 0.999),
                weight_decay=0.9)
    # # elif cfg.OPTIM.METHOD == 'SGD':
    #     return optim.SGD(params,
    #                lr=cfg.OPTIM.LR,
    #                momentum=cfg.OPTIM.MOMENTUM,
    #                dampening=cfg.OPTIM.DAMPENING,
    #                weight_decay=cfg.OPTIM.WD,
    #                nesterov=cfg.OPTIM.NESTEROV)
    # else:
    #     raise NotImplementedError