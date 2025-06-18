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
from sota.aetta2d import AETTA
from sota.img_hist_update2d import ImgHistUpdate
from sota.bn_update2d import BNUpdate
import numpy as np


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
        self.est = AETTA()
        self.con_loss = nn.KLDivLoss(reduction='batchmean')
        # self.con_loss = nn.L1Loss()
        # self.est_avg = None
        self.est_list = []
        self.imgupdate = ImgHistUpdate()
        # self.bnupdate = BNUpdate(model=model)
        

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
            est_1, est_2, est_3, est_avg, mismatch_mask, entropy, var, acc = self.est.aetta(input=x,pred=pred,model=eval_model, multi_eval=multi_eval)
        
        else:
            est_1, est_2, est_3, est_avg, mismatch_mask = self.est.aetta(input=x,pred=pred,model=eval_model, multi_eval=multi_eval)
        
        # 适应系数在整个batch上来算
        adapt_alpha = est_avg.mean()/100
        
        # adapt_alpha = 1.0
        
        # outputs = model(x)
        # standard_ema = self.model_ema(x)
        
        # 对图像进行直方图匹配，然后再把更新后的图像输入网络，得到更新后的结果
        # updated_x = self.imgupdate.img_update_histbank(x, pred, est_1, est_2, est_3, est_avg)
        # updated_x = updated_x.cuda().float()
        # outputs = model(updated_x)
        # standard_ema = self.model_ema(updated_x)
        
        
        # 对图像进行特征匹配，然后再把更新后的图像输入网络，得到更新后的结果
        feature = model.get_feature(x)
        updated_feat = self.imgupdate.img_update_featbank(feature, pred, est_1, est_2, est_3, est_avg)
        # updated_feat = feature
        
        # 这里测试使用entropy的效果
        # updated_feat = self.imgupdate.img_update_featbank(feature, pred, est_1, est_2, est_3, entropy)
        # adapt_alpha = 1.0
        
        updated_outputs = model.get_output(updated_feat)
        outputs = model.get_output(feature)
        standard_ema = self.model_ema(x)
        
        sem_loss = adapt_alpha*((softmax_entropy(outputs, updated_outputs)).mean(0) + (softmax_entropy(updated_outputs, outputs)).mean(0))/2.0
        # sem_loss = adapt_alpha*((softmax_entropy(outputs, outputs)).mean(0) + (softmax_entropy(outputs, outputs)).mean(0))/2.0
        
        # 使用表现更好的BN参数进行模型的推理
        # 以Batch为单位进行更新，因此使用est_avg.mean()，使用整个batch的一个平均表现
        # model = self.bnupdate.update(est_avg.mean(), x, model)
        # self.model_ema = self.bnupdate.update(est_avg.mean(), x, self.model_ema)
        
        # 尝试进行特征层面的对齐:此处尝试，如果一个case效果好，那么认为该case是更加贴近源域的
        # 就让该case学生模型的输出特征和源模型的输出特征尽可能保持一致
        # self.est_list.extend(est_avg.tolist())
        # print(est_avg.shape, len(self.est_list))
        # estp90 = np.percentile(self.est_list, 90)
        # print(f'estp90:{estp90}')
        
        # 做特征对齐
        # feature = model.get_feature(x)
        
        # feature = model.get_feature(x)
        # if(est_avg.mean()>estp95): 
        #     source_feature = self.source_model.get_feature(x)
        #     con_loss = self.con_loss(nn.functional.softmax(feature, dim=1).log(), nn.functional.softmax(source_feature, dim=1))
        #     # con_loss = self.con_loss(feature, source_feature)
        # else:
        #     con_loss = 0.0
        
        # 尝试进行输出层面的对齐:此处尝试，如果一个case效果好，那么认为该case是更加贴近源域的
        # 就让该case学生模型的输出特征和源模型的输出特征尽可能保持一致
        # outputs = model(x)
        # if(est_avg.mean()>estp90): 
        #     source_output = self.source_model(x)
        #     con_loss = self.con_loss(nn.functional.softmax(feature, dim=1).log(), nn.functional.softmax(source_feature, dim=1))
        #     con_loss = self.con_loss(feature, source_feature)
        #     con_loss = adapt_alpha*((softmax_entropy(outputs, source_output)).mean(0) + (softmax_entropy(source_output, outputs)).mean(0))/2.0
        #     con_loss = ((softmax_entropy(outputs, source_output)).mean(0) + (softmax_entropy(source_output, outputs)).mean(0))/2.0
        # else:
        #     con_loss = 0.0
        # standard_ema = self.model_ema(x)
        
        # outputs = model(x)
        # standard_ema = self.model_ema(x)
         
        ce_loss = ((softmax_entropy(outputs, standard_ema)).mean(0) + (softmax_entropy(standard_ema, outputs)).mean(0))/2.0
        # ce_loss = ((1-adapt_alpha)*(softmax_entropy(outputs, standard_ema)).mean(0) + (adapt_alpha)*(softmax_entropy(standard_ema, outputs)).mean(0))/2.0
        
        # 在此基础之上加上Entropy loss
        # proba = outputs.softmax(1)
        # n,c,d,h,w = outputs.size()
        # ent_loss = -adapt_alpha*(proba * torch.log2(proba + 1e-10)).sum() / \
        #     (n*d*h*w*torch.log2(torch.tensor(c, dtype=torch.float)))
        # 进行 Region_aware Entropy Loss
        # mismatch_mask = torch.tensor(mismatch_mask).unsqueeze(1)
        # mismatch_proba = proba * mismatch_mask.cuda()
        # ent_loss = -adapt_alpha*(mismatch_proba * torch.log(mismatch_proba + 1e-10)).sum() / \
        #     (mismatch_mask.sum()+ 1e-10)
        
        # 对每一个图像，分别去做，尝试进行输出层面的对齐:此处尝试，如果一个case效果好，那么认为该case是更加贴近源域的
        # 就让该case学生模型的输出特征和源模型的输出特征尽可能保持一致
        # 只对 match 的区域去做
        # source_proba = self.source_model(x).softmax(1)
        # con_loss = 0.0
        # count = 0
        # match_mask = ~mismatch_mask
        # for i in range(est_avg.shape[0]):
        #     if(est_avg[i]>estp90): 
        #         count+=1
        #         match_prob = proba[i] * match_mask[i].cuda()
        #         match_source_prob = source_proba[i] * match_mask[i].cuda()
        #         # con_loss = adapt_alpha*((softmax_entropy(outputs, source_output)).mean(0) + (softmax_entropy(source_output, outputs)).mean(0))/2.0
        #         con_loss += -(match_prob * torch.log(match_source_prob + 1e-10)).sum() / \
        #             (match_mask.sum()+ 1e-10)
        # con_loss = con_loss/(count+1e-6)
        # ce_loss = 0.0
        loss = ce_loss + sem_loss
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
