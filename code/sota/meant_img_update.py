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
from sota.aetta import AETTA
from sota.img_hist_update import ImgHistUpdate

# dicece_loss = DiceCELoss(lambda_dice=0)
class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, anchor_model,optimizer, steps=2, episodic=False, mt_alpha=0.99, rst_m=0.1,):
        super().__init__()
        self.model = model
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
        self.est = AETTA()
        self.imgupdate = ImgHistUpdate()
        self.imgupdate_2 = ImgHistUpdate()
        self.imgupdate_3 = ImgHistUpdate()
        self.imgupdate_4 = ImgHistUpdate()
        

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
   
    torch.autograd.set_detect_anomaly(True)
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        pred = self.forward_no_adapt(x)
        pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
        eval_model = deepcopy(model)

        est_WT, est_TC, est_ET, est_avg, mismatch_mask, entropy = self.est.aetta(input=x,pred=pred,model=eval_model)
        
        # 对图像进行直方图匹配，然后再把更新后的图像输入网络，得到更新后的结果
        # updated_x = self.imgupdate.img_update(x, pred, est_WT, est_TC, est_ET)
        # updated_x = updated_x.cuda().float()
        # outputs = model(updated_x)
        # standard_ema = self.model_ema(updated_x)
        
        # outputs = model(x)
        # standard_ema = self.model_ema(x)
        
        adapt_alpha = est_avg/100
        # adapt_alpha = 0.99
        
        # 对图像进行特征匹配，然后再把更新后的图像输入网络，得到更新后的结果
        feature = model.get_feature(x)
        # 只对最后一层的特征进行更新
        # updated_feat_1 = self.imgupdate.img_update_featbank(feature[-1], pred, est_WT, est_TC, est_ET, est_avg)
        # updated_feat_2 = self.imgupdate_2.img_update_featbank(feature[-2], pred, est_WT, est_TC, est_ET, est_avg)
        updated_feat_3 = self.imgupdate_3.img_update_featbank(feature[-3], pred, est_WT, est_TC, est_ET, est_avg)
        # 测试用entropy
        # updated_feat_3 = self.imgupdate_3.img_update_featbank(feature[-3], pred, est_WT, est_TC, est_ET, 1-entropy)
        # updated_feat_4 = self.imgupdate_4.img_update_featbank(feature[-4], pred, est_WT, est_TC, est_ET, est_avg)
        updated_outputs = model.get_output(feature[:-3]+[updated_feat_3]+[feature[-2]]+[feature[-1]])
        outputs = model.get_output(feature)
        standard_ema = self.model_ema(x)
       
        sem_loss = 100*adapt_alpha*((softmax_entropy(outputs, updated_outputs)).mean(0) + (softmax_entropy(updated_outputs, outputs)).mean(0))/2.0
        # loss = (softmax_entropy(outputs, standard_ema)).mean(0) 
        ce_loss = ((softmax_entropy(outputs, standard_ema)).mean(0) + (softmax_entropy(standard_ema, outputs)).mean(0))/2.0
        # ce_loss = ((1-adapt_alpha)*(softmax_entropy(outputs, standard_ema)).mean(0) + (adapt_alpha)*(softmax_entropy(standard_ema, outputs)).mean(0))/2.0
        # 在此基础之上加上Entropy loss
        proba = outputs.softmax(1)
        # n,c,d,h,w = outputs.size()
        # ent_loss = -adapt_alpha*(proba * torch.log2(proba + 1e-10)).sum() / \
        #     (n*d*h*w*torch.log2(torch.tensor(c, dtype=torch.float)))
        # 进行 Region_aware Entropy Loss
        # mismatch_mask = torch.tensor(mismatch_mask).unsqueeze(0).unsqueeze(0).expand(1, 4, 128, 128, 128)
        # mismatch_proba = proba[mismatch_mask]
        # ent_loss = -adapt_alpha*(mismatch_proba * torch.log(mismatch_proba + 1e-10)).sum() / \
        #     (mismatch_mask.sum()+ 1e-10)
        
        # sem_loss = 0.0
        loss = ce_loss +  sem_loss
        print(f'ce_loss:{ce_loss}, sem_loss:{sem_loss}, loss:{loss}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=adapt_alpha)
        # return self.model_ema(updated_x)
        # return model(updated_x)
        # return model(updated_x)
        return model(x)


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, d, h, w =  x.shape
    # for KD:
    # x_ema = x_ema/2.
    # x = x/2.
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * d * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
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
        ema_param.data[:] = (1 - alpha_teacher) * ema_param[:].data[:] + (alpha_teacher) * param[:].data[:]
    return ema_model


def configure_cotta_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model
