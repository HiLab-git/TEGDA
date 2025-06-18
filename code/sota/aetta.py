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

class AETTA(nn.Module):
    def __init__(self):
        super(AETTA, self).__init__()
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
        # 对熵值进行归一化后，再用于权值的计算，当最大值最小值相等时，就等于0.5
        ent_pred = (ent_pred-ent_pred.min()+1e-6)/(ent_pred.max()-ent_pred.min()+2e-6)
        # 使用熵理论上的最大值进行归一化，即soft=0.5，一半预测有，一半预测没有，而非本身的最大最小值
        # ENT_MAX = 0.6931
        # ent_pred = ent_pred/ENT_MAX 
        ent_weight = ((1-ent_pred)[uni_pred==1].sum()+1e-6)/(uni_pred.sum()+1e-6)
        return ent_weight ** alpha
    
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
                pred = net.forward_no_adapt(input)  # batch_size, n_classes
                pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        predictions = torch.stack(predictions, dim=1)  # (1,10,4,128,128,128)
        pred = torch.argmax(predictions, dim=2)[0].cpu().numpy()  # (10,128,128,128)
        mean = torch.mean(predictions, dim=1) #(1,4,128,128,128)
        # mean_pred_class = torch.argmax(mean_pred, dim=1)
        # std = torch.std(predictions, dim=1)

        # conf_mean = mean[:, curr_pred].diagonal()
        # conf_std = std[:, curr_pred].diagonal()
        # mean_for_curr_pred = conf_mean.mean()
        # std_for_curr_pred = conf_std.mean()

        total_avg_softmax = torch.mean(mean, dim=[0,2,3,4])
        e_avg = (-total_avg_softmax * torch.log(total_avg_softmax + 1e-6)).sum()
        
        # 找出不确定性高的区域
        WT_region = self.calculate_ent_region(label_list=[1,2,3],pred=pred,ent_threshold=0.01)
        TC_region = self.calculate_ent_region(label_list=[2,3],pred=pred,ent_threshold=0.01)
        ET_region = self.calculate_ent_region(label_list=[3],pred=pred,ent_threshold=0.01)
        
        # Prediction disagreement with dropouts,这里是通过acc来计算，但是这其实并不适用于分割任务
        # match_ratio = (curr_pred_tensor.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,n_iter,1,1,1) == pred).sum(dim=1, dtype=float) / n_iter
        # acc = match_ratio.mean()
        # 以原本的预测为gt，dropout后的一系列预测为seg，去算Dice
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
            # 针对高不确定性的区域计算Dice
            # per_dropout_WT_dice = get_multi_class_evaluation_score(s_volume=pred[i][WT_region],g_volume=curr_pred[WT_region],label_list=[1,2,3],\
            #     fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            # per_dropout_TC_dice = get_multi_class_evaluation_score(s_volume=pred[i][TC_region],g_volume=curr_pred[TC_region],label_list=[2,3],\
            #     fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            # per_dropout_ET_dice = get_multi_class_evaluation_score(s_volume=pred[i][ET_region],g_volume=curr_pred[ET_region],label_list=[3],\
            #     fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
            WT_dice_list.append(per_dropout_WT_dice)
            TC_dice_list.append(per_dropout_TC_dice)
            ET_dice_list.append(per_dropout_ET_dice)
        aetta_WT_dice = np.mean(WT_dice_list)
        aetta_TC_dice = np.mean(TC_dice_list)
        aetta_ET_dice = np.mean(ET_dice_list)
        # return acc.item(), mean_for_curr_pred.item(), std_for_curr_pred.item(), e_avg.item()
        return aetta_WT_dice.item(),aetta_TC_dice.item(),aetta_ET_dice.item(),e_avg.item()

    
    def evaluate_dropout_2(self, input, curr_pred, net, n_iter=10, dropout=0.5):
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
        
        # 确定不同类别的权重
        WT_weight = self.calculate_ent_weight(label_list=[1,2,3],pred=pred,alpha=1.0)
        TC_weight = self.calculate_ent_weight(label_list=[2,3],pred=pred,alpha=1.0)
        ET_weight = self.calculate_ent_weight(label_list=[3],pred=pred,alpha=1.0)
        
        # Prediction disagreement with dropouts,这里是通过acc来计算，但是这其实并不适用于分割任务
        # match_ratio = (curr_pred_tensor.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,n_iter,1,1,1) == pred).sum(dim=1, dtype=float) / n_iter
        # acc = match_ratio.mean()
        # 以原本的预测为gt，dropout后的一系列预测为seg，去算Dice
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
        aetta_WT_dice = WT_weight*np.mean(WT_dice_list)
        aetta_TC_dice = TC_weight*np.mean(TC_dice_list)
        aetta_ET_dice = ET_weight*np.mean(ET_dice_list)
        
        # return acc.item(), mean_for_curr_pred.item(), std_for_curr_pred.item(), e_avg.item()
        return aetta_WT_dice.item(),aetta_TC_dice.item(),aetta_ET_dice.item(), mismatch_mask, mean
    
    def aetta_1(self, input, pred, model):
        est_WT,est_TC,est_ET, e_avg = self.evaluate_dropout(input, pred, model, dropout=0.4)
        # est_WT,est_TC,est_ET = self.evaluate_dropout(input, pred, model, dropout=0.4)
        est_WT = round(est_WT*100,2)
        est_TC = round(est_TC*100,2)
        est_ET = round(est_ET*100,2)
        # acc_est_json['est_dropout'] += [est_acc]
        # acc_est_json['est_dropout_avg_entropy'] += [e_avg]
        # acc_est_json['est_dropout_softmax_mean'] += [mean]
        # acc_est_json['est_dropout_softmax_std'] += [std]

        # est_err = 1 - est_acc
        if self.est_WT_ema is None and self.est_TC_ema is None and self.est_ET_ema is None:
            self.est_WT_ema = est_WT
            self.est_TC_ema = est_TC
            self.est_ET_ema = est_ET
        
        
        MAX_ENTROPY = 1.386
        N_CLASS = 4
        
        updated_WT = est_WT * (e_avg / MAX_ENTROPY)
        updated_TC = est_TC * (e_avg / MAX_ENTROPY)
        updated_ET = est_ET * (e_avg / MAX_ENTROPY)
        # updated = est_err / (e_avg / MAX_ENTROPY) ** 3
        # updated = max(0., min(1. - 1. / N_CLASS, updated))

        est_WT = self.est_WT_ema * 0.6 + updated_WT * 0.4
        est_TC = self.est_TC_ema * 0.6 + updated_TC * 0.4
        est_ET = self.est_ET_ema * 0.6 + updated_ET * 0.4
        self.est_WT_ema = est_WT
        self.est_TC_ema = est_TC
        self.est_ET_ema = est_ET

        # # acc_est_json['aetta'] += [100 * (1. - updated)]
        
        # final_acc = 100 * (1. - updated)
        
        return est_WT, est_TC, est_ET
    
    def aetta(self, input, pred, model):
        est_WT,est_TC,est_ET, mismatch_mask, pred_mean= self.evaluate_dropout_2(input, pred, model, dropout=0.4)
        # est_WT,est_TC,est_ET = self.evaluate_dropout(input, pred, model, dropout=0.4)
        est_WT = round(est_WT*100,2)
        est_TC = round(est_TC*100,2)
        est_ET = round(est_ET*100,2)
        est_avg = round((est_WT+est_ET+est_TC)/3,2)
        # acc_est_json['est_dropout'] += [est_acc]
        # acc_est_json['est_dropout_avg_entropy'] += [e_avg]
        # acc_est_json['est_dropout_softmax_mean'] += [mean]
        # acc_est_json['est_dropout_softmax_std'] += [std]

        # est_err = 1 - est_acc
        # if self.est_WT_ema is None and self.est_TC_ema is None and self.est_ET_ema is None:
        #     self.est_WT_ema = est_WT
        #     self.est_TC_ema = est_TC
        #     self.est_ET_ema = est_ET
        
        
        # est_WT = 0.6 * self.est_WT_ema + 0.4 * est_WT
        # est_TC = 0.6 * self.est_TC_ema + 0.4 * est_TC
        # est_ET = 0.6 * self.est_ET_ema + 0.4 * est_ET
        entropy = -np.sum(pred_mean.cpu().numpy()*np.log(pred_mean.cpu().numpy()+1e-10),axis=1).mean()
        
        return est_WT, est_TC, est_ET, est_avg, mismatch_mask, entropy