from copy import deepcopy
import math
from xml.etree.ElementInclude import FatalIncludeError
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import torch
import torch.nn as nn
import torch.jit
from monai.losses import DiceLoss, DiceCELoss
import random
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
from utils.utils import rotate_single_random,derotate_single_random,add_gaussian_noise_3d
from robustbench.losses import WeightedCrossEntropyLoss,DiceCeLoss,DiceLoss,center_alignment_loss,KDLoss,mmd_loss
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
dicece_loss = DiceCeLoss(4)
from sota.adic2d import ADIC
import numpy as np
from sota.img_update import *

# dicece_loss = DiceCELoss(lambda_dice=0)
class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, anchor_model,optimizer, steps=2, episodic=False, mt_alpha=0.99, rst_m=0.1,):
        super().__init__()
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.optimizer = optimizer
        self.model_ema = anchor_model
        # self.rec_loss = nn.MSELoss()
        # self.model_state, self.model_ema,  = \
        #     copy_model_and_optimizer(self.model, self.optimizer)
        self.num_classes = 4 
        self.mt = mt_alpha
        self.rst = rst_m
        self.model = model
        self.source_model = deepcopy(model)
        for param in self.source_model.parameters():
            param.requires_grad = False
        self.est = ADIC()
        self.con_loss = nn.KLDivLoss(reduction='batchmean')
        # self.con_loss = nn.L1Loss()
        # self.est_avg = None
        self.est_list = []
        self.imgupdate = ImgUpdate()
        

    # def forward(self, x, label_batch, names):
    def forward(self, x):
        # self.label = label_batch
        
        if self.episodic:
            self.reset()
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs
    
    @torch.no_grad()
    def forward_no_adapt(self, x):
        outputs = self.model(x)
        return outputs
    
    def get_index(self):
        return self.index
   
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, multi_eval=True):
        pred = self.forward_no_adapt(x)
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        eval_model = deepcopy(model)
        
        if(multi_eval):
            est_1, est_2, est_3, est_avg, mismatch_mask, entropy, var, acc = self.est.ADIC(input=x,pred=pred,model=eval_model, multi_eval=multi_eval)
        else:
            est_1, est_2, est_3, est_avg, mismatch_mask = self.est.ADIC(input=x,pred=pred,model=eval_model, multi_eval=multi_eval)
        
        adapt_alpha = est_avg.mean()/100
 
        feature = model.get_feature(x)
        updated_feat = self.imgupdate.img_update_featbank(feature, pred, est_1, est_2, est_3, est_avg)
        # updated_feat = feature
        
        updated_outputs = model.get_output(updated_feat)
        outputs = model.get_output(feature)
        standard_ema = self.model_ema(x)
        
        sem_loss = adapt_alpha*((softmax_entropy(outputs, updated_outputs)).mean(0) + (softmax_entropy(updated_outputs, outputs)).mean(0))/2.0
        # standard_ema = self.model_ema(x)
         
        ce_loss = ((softmax_entropy(outputs, standard_ema)).mean(0) + (softmax_entropy(standard_ema, outputs)).mean(0))/2.0
        loss = ce_loss + sem_loss
        print(f'ce_loss:{ce_loss}, sem_loss:{sem_loss}, loss:{loss}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=adapt_alpha)
        return model(x)


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    # for KD:
    # x_ema = x_ema/2.
    # x = x/2.
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if True:#isinstance(m, nn.BatchNorm2d): collect all 
        if 'dec1.last' not in nm:
            print(nm, '55',m, '496')
            for np, p in m.named_parameters():
                
                if np in ['weight', 'bias'] and p.requires_grad:
                # if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

    
def update_ema_variables(ema_model, model, alpha_teacher):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    # for ema_param, param in zip(ema_model.dec1.parameters(), model.dec1.parameters()):
        # ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def configure_model(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
