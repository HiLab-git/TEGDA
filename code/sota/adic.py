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

class ADIC(nn.Module):
    def __init__(self):
        super(ADIC, self).__init__()
        self.est_WT_ema = None
        self.est_TC_ema = None
        self.est_ET_ema = None
        
    def calculate_ent_region(self, label_list, pred, ent_threshold):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)
        soft_pred = np.mean(pred_sub, axis=0)
        ent_pred = (-soft_pred * np.log(soft_pred + 1e-6))
        target_region = np.where(ent_pred-ent_threshold > 0)
        
        return target_region
    
    def calculate_ent_weight(self, label_list, pred, alpha=1.0):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)
        soft_pred = np.mean(pred_sub, axis=0)
        uni_pred = np.max(pred_sub,axis=0)
        ent_pred = -(soft_pred * np.log(soft_pred + 1e-6) + (1-soft_pred)*np.log(1-soft_pred+1e-6))
        ent_pred = (ent_pred-ent_pred.min()+1e-6)/(ent_pred.max()-ent_pred.min()+2e-6)
        ent_weight = ((1-ent_pred)[uni_pred==1].sum()+1e-6)/(uni_pred.sum()+1e-6)
        return ent_weight ** alpha

    
    def evaluate_dropout(self, input, curr_pred, net, n_iter=10, dropout=0.5):
        # Dropout inference sampling
        predictions = []
        input = input.cuda().float()
        # curr_pred_tensor = torch.tensor(curr_pred).cuda()
        for module in net.modules():
            if isinstance(module,nn.Dropout):
                module.train()
        with torch.no_grad():
            for _ in range(n_iter):
                pred = net.forward_no_adapt(input)  # batch_size, n_classes
                pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (1,10,4,128,128,128)
        pred = torch.argmax(predictions, dim=2)[0].cpu().numpy()  # (10,128,128,128)
        mean = torch.mean(predictions, dim=1) #(1,4,128,128,128)
        
        avg_pred = torch.argmax(mean,dim=1)[0].cpu().numpy()
        mismatch_mask = (curr_pred != avg_pred)
        
        WT_weight = self.calculate_ent_weight(label_list=[1,2,3],pred=pred,alpha=1.0)
        TC_weight = self.calculate_ent_weight(label_list=[2,3],pred=pred,alpha=1.0)
        ET_weight = self.calculate_ent_weight(label_list=[3],pred=pred,alpha=1.0)
        
        WT_dice_list = []
        TC_dice_list = []
        ET_dice_list = []
        for i in range(n_iter):
            per_dropout_WT_dice = get_multi_class_evaluation_score(s_volume=pred[i],g_volume=curr_pred,label_list=[1,2,3],\
                fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            per_dropout_TC_dice = get_multi_class_evaluation_score(s_volume=pred[i],g_volume=curr_pred,label_list=[2,3],\
                fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            per_dropout_ET_dice = get_multi_class_evaluation_score(s_volume=pred[i],g_volume=curr_pred,label_list=[3],\
                fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            WT_dice_list.append(per_dropout_WT_dice)
            TC_dice_list.append(per_dropout_TC_dice)
            ET_dice_list.append(per_dropout_ET_dice)
        ADIC_WT_dice = WT_weight*np.mean(WT_dice_list)
        ADIC_TC_dice = TC_weight*np.mean(TC_dice_list)
        ADIC_ET_dice = ET_weight*np.mean(ET_dice_list)
    
        return ADIC_WT_dice.item(),ADIC_TC_dice.item(),ADIC_ET_dice.item(), mismatch_mask, mean
    
    
    def ADIC(self, input, pred, model):
        est_WT,est_TC,est_ET, mismatch_mask, pred_mean= self.evaluate_dropout(input, pred, model, dropout=0.4)
        est_WT = round(est_WT*100,2)
        est_TC = round(est_TC*100,2)
        est_ET = round(est_ET*100,2)
        est_avg = round((est_WT+est_ET+est_TC)/3,2)
        
        entropy = -np.sum(pred_mean.cpu().numpy()*np.log(pred_mean.cpu().numpy()+1e-10),axis=1).mean()
        
        return est_WT, est_TC, est_ET, est_avg, mismatch_mask, entropy