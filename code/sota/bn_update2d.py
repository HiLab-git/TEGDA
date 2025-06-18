from collections import deque
from typing import Union, Tuple, Optional

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms.functional import to_pil_image, rotate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import copy

def get_stats(net):
    mean, var = [], []
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            mean.append(m.running_mean.clone().detach())
            var.append(m.running_var.clone().detach())
    return mean, var

def update_stats(net, mean, var):
    count = 0
    with torch.no_grad():
        for nm, m in net.named_modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.running_mean = mean[count].clone().detach().cuda()
                m.running_var = var[count].clone().detach().cuda()
                count += 1
                
def test_no_adapt(net, image):
    image = image.cuda()
    image = image.float()
    with torch.no_grad():
        y1 = net.forward_no_adapt(image)
        y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()[0] 
    
    return label
        
        
class BNUpdate():
    def __init__(self, model):
        self.est_ema = None
        self.ori_mean, self.ori_std = get_stats(model)
        self.val_mean = None
        self.val_std = None
    
    def update(self, est_avg, input, model):
        # 针对full image
        # 只计算大脑的hist
        update_stats(model, self.ori_mean, self.ori_std)
        model.train()
        with torch.no_grad():
            model(input)
        curr_val_mean, curr_val_std = get_stats(model)
        
        if self.est_ema == None:
            self.est_ema = est_avg
            self.val_mean = curr_val_mean
            self.val_std = curr_val_std
        
        if(est_avg<self.est_ema):
            alpha = 0.5
            beta = 0.5
            update_model = True
        else:
            alpha = 0.9
            beta = 0.1
            update_model = False
        
        updated_val_mean = []
        updated_val_std = []
        for mean_a, std_a, mean_b, std_b in zip(self.val_mean, self.val_std, curr_val_mean, curr_val_std):
            updated_val_mean.append(alpha * mean_a + beta * mean_b)
            updated_val_std.append(alpha * std_a + beta * std_b)

        updated_est = 0.9 * self.est_ema + 0.1 * est_avg
        
        if(update_model):
            update_stats(model, updated_val_mean, updated_val_std)
        
        self.val_mean = updated_val_mean
        self.val_std = updated_val_std
        self.est_ema = updated_est
        
        return model