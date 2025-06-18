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
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from tqdm import tqdm
from copy import deepcopy
import numpy as np

class AETTA4Seg(nn.Module):
    def __init__(self):
        self.est_ema_dropout = None
        
        
    def evaluate_dropout(self, input, curr_pred, net, n_iter=10, dropout=0.5):
        # Dropout inference sampling
        predictions = []
        input = input.cuda()
        # curr_pred_tensor = torch.tensor(curr_pred).cuda()
        for module in net.modules():
            if isinstance(module,nn.Dropout):
                module.train()
        with torch.no_grad():
            for _ in range(n_iter):
                pred = net.forward_dropout(input)  # batch_size, n_classes
                pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # batch_size, n_iter, n_classes
        pred = torch.argmax(predictions, dim=2)[0].cpu().numpy()
        mean = torch.mean(predictions, dim=1)
        std = torch.std(predictions, dim=1)
        img_level_unc = std.sum().cpu()
        
        # 因为image-level的uncertainty会因为目标的大小而产生变化
        ent_pred = (-mean * torch.log(mean + 1e-6)).cpu()
        ent_threshold = 0.2
        est_size = torch.where(ent_pred-ent_threshold > 0, torch.tensor(1.0), torch.tensor(0.0)).sum()
        if est_size>0:
            img_level_unc = img_level_unc/est_size
        else:
            img_level_unc = 1.0
        
        #对算出来的不确定性进行归一化
        # img_level_unc = (img_level_unc-img_level_unc.min())/(img_level_unc.max()-img_level_unc.min())
        
        # final_acc = 1-img_level_unc
        
        return img_level_unc
    
    
    def get_score(self, input, pred, model):
        final_acc = self.evaluate_dropout(input, pred, model, dropout=0.4)
        
        return final_acc