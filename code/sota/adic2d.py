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
        soft_pred = np.mean(pred_sub, axis=1)
        uni_pred = np.max(pred_sub,axis=1)
        ent_pred = -(soft_pred * np.log(soft_pred + 1e-6) + (1-soft_pred)*np.log(1-soft_pred+1e-6))
        ent_pred = (ent_pred-ent_pred.min()+1e-6)/(ent_pred.max()-ent_pred.min()+2e-6)
        # ENT_MAX = 0.6931
        # ent_pred = ent_pred/ENT_MAX 
        ent_weight = []
        ent_weight = ((np.sum((1-ent_pred)*uni_pred,axis=(1,2))+1e-6)/(np.sum(uni_pred, axis=(1,2))+1e-6))
        return ent_weight
    
    def calculate_ent_weight_inv(self, label_list, pred, alpha=1.0):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)
        soft_pred = np.mean(pred_sub, axis=1)
        uni_pred = np.max(pred_sub,axis=1)
        ent_pred = -(soft_pred * np.log(soft_pred + 1e-6) + (1-soft_pred)*np.log(1-soft_pred+1e-6))
        ent_pred = (ent_pred-ent_pred.min()+1e-6)/(ent_pred.max()-ent_pred.min()+2e-6)
        ent_weight = []
        ent_weight = ((np.sum((ent_pred)*uni_pred,axis=(1,2))+1e-6)/(np.sum(uni_pred, axis=(1,2))+1e-6))
        return ent_weight
    
    def get_batch_dice(self, s, g, label_list):
        s_sub = np.zeros_like(s)
        g_sub = np.zeros_like(g)
        for lab in label_list:
            s_sub = s_sub + np.asarray(s == lab, np.uint8)
            g_sub = g_sub + np.asarray(g == lab, np.uint8)
        s = np.asarray(s > 0, np.uint8)
        g = np.asarray(g > 0, np.uint8)
        assert(len(s.shape)== len(g.shape))
        prod = np.multiply(s, g)
        s0 = prod.sum(axis=(1,2))
        s1 = s.sum(axis=(1,2))
        s2 = g.sum(axis=(1,2))
        dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
        return dice
    
    def get_batch_acc(self, s, g, label_list):
        correct_pixels = np.zeros(s.shape[0], dtype=np.float32)  
        total_pixels = s.shape[1] * s.shape[2]  
        for lab in label_list:
            s_sub = np.asarray(s == lab, np.uint8)
            g_sub = np.asarray(g == lab, np.uint8)
            
            correct_pixels += np.sum(s_sub == g_sub, axis=(1, 2))
        
        accuracy = correct_pixels / total_pixels
        return accuracy

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
        predictions = torch.stack(predictions, dim=1)  # (24,10,4,320,320)
        pred = torch.argmax(predictions, dim=2).cpu().numpy()  # (24,10,320,320)
        mean = torch.mean(predictions, dim=1) #(24,4,320,320)
        
        avg_pred = torch.argmax(mean,dim=1).cpu().numpy()
        mismatch_mask = (curr_pred != avg_pred)
        
        est_1_weight = self.calculate_ent_weight(label_list=[1],pred=pred,alpha=1.0)
        est_2_weight = self.calculate_ent_weight(label_list=[2],pred=pred,alpha=1.0)
        est_3_weight = self.calculate_ent_weight(label_list=[3],pred=pred,alpha=1.0)
        
        est_1_weight_inv = self.calculate_ent_weight_inv(label_list=[1],pred=pred,alpha=1.0)
        est_2_weight_inv = self.calculate_ent_weight_inv(label_list=[2],pred=pred,alpha=1.0)
        est_3_weight_inv = self.calculate_ent_weight_inv(label_list=[3],pred=pred,alpha=1.0)
        
        cls1_dice_list = []
        cls2_dice_list = []
        cls3_dice_list = []
        for i in range(n_iter):
            per_dropout_1_dice = self.get_batch_dice(s = pred[:,i],g = curr_pred,label_list = [1])
            per_dropout_2_dice = self.get_batch_dice(s=pred[:,i],g=curr_pred,label_list = [2])
            per_dropout_3_dice = self.get_batch_dice(s=pred[:,i],g=curr_pred,label_list = [3])
            cls1_dice_list.append(per_dropout_1_dice)
            cls2_dice_list.append(per_dropout_2_dice)
            cls3_dice_list.append(per_dropout_3_dice)
        
        est_avg_noweight = ((np.mean(cls1_dice_list)+np.mean(cls2_dice_list)+np.mean(cls3_dice_list))/3)*100
        est_avg_weight_inv = ((est_1_weight_inv*np.mean(cls1_dice_list)+
                               est_2_weight_inv*np.mean(cls2_dice_list)+
                               est_3_weight_inv*np.mean(cls3_dice_list))/3)*100
        
        ADIC_WT_dice = est_1_weight*np.mean(cls1_dice_list)
        ADIC_TC_dice = est_2_weight*np.mean(cls2_dice_list)
        ADIC_ET_dice = est_3_weight*np.mean(cls3_dice_list)
        
        est_WT = ADIC_WT_dice*100
        est_TC = ADIC_TC_dice*100
        est_ET = ADIC_ET_dice*100
        est_avg = (est_WT+est_ET+est_TC)/3
        
        # return acc.item(), mean_for_curr_pred.item(), std_for_curr_pred.item(), e_avg.item()
        return ADIC_WT_dice,ADIC_TC_dice,ADIC_ET_dice,est_avg, mismatch_mask, predictions, mean, est_avg_noweight,est_avg_weight_inv
    
    def get_acc(self, predictions,curr_pred,entropy,n_iter=10):
        pred = torch.argmax(predictions, dim=2).cpu().numpy()  # (24,10,320,320)
        WT_dice_list = []
        TC_dice_list = []
        ET_dice_list = []
        for i in range(n_iter):
            per_dropout_1_acc = self.get_batch_acc(s=pred[:,i],g = curr_pred,label_list = [1])
            per_dropout_2_acc = self.get_batch_acc(s=pred[:,i],g=curr_pred,label_list = [2])
            per_dropout_3_acc = self.get_batch_acc(s=pred[:,i],g=curr_pred,label_list = [3])
            WT_dice_list.append(per_dropout_1_acc)
            TC_dice_list.append(per_dropout_2_acc)
            ET_dice_list.append(per_dropout_3_acc)
        
        acc = (np.mean(WT_dice_list)+np.mean(TC_dice_list)+np.mean(ET_dice_list))/3
        
        return acc
        
    def ADIC(self, input, pred, model, multi_eval=True):
        est_WT,est_TC,est_ET,est_avg, mismatch_mask, predictions, pred_mean, est_avg_noweight,est_avg_weight_inv \
            = self.evaluate_dropout(input, pred, model, dropout=0.4)

        if(multi_eval):
            entropy = -np.sum(pred_mean.cpu().numpy()*np.log(pred_mean.cpu().numpy()+1e-10),axis=1)
            norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
            mean_norm_entropy = np.mean(norm_entropy,axis=(1,2))
            
            variance = (np.var(predictions.cpu().numpy(),axis=1)*mismatch_mask[:,None,:,:])
            norm_variance = (variance - variance.min()) / (variance.max()-variance.min())
            mean_norm_variance = norm_variance.mean()
            
            acc = self.get_acc(predictions,pred,mean_norm_entropy.mean())
            return est_WT, est_TC, est_ET, est_avg, mismatch_mask, mean_norm_entropy, mean_norm_variance, acc
        
        
        return est_WT, est_TC, est_ET, est_avg, mismatch_mask
    