import logging
from statistics import mode
import time
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from sota import tent
from sota import norm
from sota import cotta
from sota import wjh01_0000
from sota import meant
from sota import meant_img_update
from sota import memo
from sota import upl
from sota import sar
from sota import vida
from sota import svdp
from sota import sitta
from sota import vdptta
from sota import vptta
from sota import vdptta_2
from sota import wesam
from sota import zff01
from utils.sam import SAM
import SimpleITK as sitk
import math

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def setup_sar(model):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(4))

    return adapt_model

def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    # logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    print(stat_names)
    # logger.info(f"stats for adaptation: %s", stat_names)
   
    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer)
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
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer)
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
    cotta_model = meant.TTA(model, anchor_model, optimizer)
    return cotta_model

def setup_meantimgupdate(model):
    anchor_model = deepcopy(model)
    # model.train()
    # anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001,betas=(0.5,0.999))
    mt_model = meant_img_update.TTA(model, anchor_model, optimizer)
    return mt_model

def setup_vsdp(model):
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)
    vsdp_model = svdp.TTA(model,anchor,anchor_model,ema_model)
    return vsdp_model

def setup_vdptta(model):
    prompt = vdptta.configure_prompt(size=128)
    # logging.info(f"prompt for adaptation: %s", prompt.parameters)
    # Froze the source model. Only optimize the prompt
    params, param_names = vdptta.collect_params(prompt)
    model.eval()
    optimizer = setup_optimizer(params)
    logging.info(f"params for adaptation: %s", param_names)
    logging.info(f"optimizer for adaptation: %s", optimizer)
    vdptta_model = vdptta.VDPTTA(prompt,model,optimizer)
    
    return vdptta_model

def setup_vdptta_2(model):
    prompt = vdptta_2.configure_prompt()
    # logging.info(f"prompt for adaptation: %s", prompt.parameters)
    # Froze the source model. Only optimize the prompt
    # params, param_names = vdptta_2.collect_params(prompt)
    model.eval()
    optimizer = optim.Adam(prompt.parameters(),
                lr=0.02,
                betas=(0.9, 0.999),
                weight_decay=0.9)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    vdptta_model = vdptta_2.VDPTTA_2(prompt,model,optimizer)
    
    return vdptta_model

def setup_zff01(model):
    prompt = zff01.configure_prompt()
    # logging.info(f"prompt for adaptation: %s", prompt.parameters)
    # Froze the source model. Only optimize the prompt
    params, param_names = zff01.collect_params(prompt)
    model.eval()
    optimizer = setup_optimizer(params)
    logging.info(f"params for adaptation: %s", param_names)
    logging.info(f"optimizer for adaptation: %s", optimizer)
    zff01_model = zff01.zff01(prompt,model,optimizer)
    
    return zff01_model


def setup_vptta(model):
    prompt = vptta.Prompt(prompt_alpha=0.03, image_size=128).to('cuda:0')
    optimizer = torch.optim.Adam(
                prompt.parameters(),
                lr=0.05,
                betas=(0.9,0.99),
                weight_decay=0.0
            )

    model = vptta.VPTTA(model, optimizer, prompt)
    
    return model

def setup_memo(model):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    cotta_model = memo.TTA(model,  optimizer )
    return cotta_model

# def setup_vida(args, model):
#     model = vida.configure_model(model, cfg)
#     model_param, vida_param = vida.collect_params(model)
#     optimizer = setup_optimizer_vida(model_param, vida_param, 5e-07, 2e-07)
#     vida_model = vida.ViDA(model, optimizer,
#                            steps=cfg.OPTIM.STEPS,
#                            episodic=cfg.MODEL.EPISODIC,
#                            unc_thr = args.unc_thr,
#                            ema = cfg.OPTIM.MT,
#                            ema_vida = cfg.OPTIM.MT_ViDA,
#                            )
#     return vida_model


def setup_upl(model):
    model.train()
    num_dec = 4
    dec_list = []
    for i in range(1, num_dec+1):
        print(model)
        dec_i = deepcopy(model.dec1)
        setattr(dec_i, 'name', f'dec_{i}')  # 修改属性或添加其他标识符
        dec_list.append(dec_i)
    optimizer_params = []
    # for dec_i in dec_list:
    #     optimizer_params.extend(dec_i.parameters())
    optimizer_params.extend(model.enc.parameters())
    optimizer = setup_optimizer(optimizer_params)
    upl_model = upl.TTA(model.enc, dec_list, optimizer
                           )
    return upl_model

def setup_sitta(model):
    cotta_model = sitta.TTA(model, 
                            repeat_num = 1,
                            check_p = '/mnt/data1/ZhouFF/TTA4MIS/model/BraTs2023_Fully_Supervised_GLI/unet_3D/iter_20000_dice_0.7799.pth'
                           )
    return cotta_model

def setup_wesam(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999))
    wesam_model = wesam.TTA(model, anchor_model, optimizer)
    
    return wesam_model

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