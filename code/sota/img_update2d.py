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

def calculate_histogram(image, hist_range = (-3,3)):
    img_fore = image[image>-0.95]
    hist, bin_edge = np.histogram(img_fore.flatten(), bins=256, range=hist_range, density=True)
    return hist, bin_edge

def calculate_cdf(hist):
    # 计算累积分布函数
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # 归一化
    return cdf_normalized

def modal_histogram_matching(target, target_hist, ref_hist, range = (-3,3)): 
    # 计算源图像和目标图像的CDF
    ref_cdf = calculate_cdf(ref_hist)
    target_cdf = calculate_cdf(target_hist)
    
    # 创建映射表
    mapping = np.interp(target_cdf, ref_cdf, np.linspace(range[0], range[1], ref_cdf.size))
    
    target_foreground = target[target>-0.95]
    matched_foreground = np.interp(target_foreground.flatten(), np.linspace(range[0], range[1], ref_cdf.size), mapping)
    
    matched_modality = copy.deepcopy(target)
    matched_modality[target>-0.95] = torch.tensor(matched_foreground).float()
    
    return matched_modality


def get_3d_crop_bounding_box(mask, margin = [5,10,10]):
    D, H, W = mask.shape
    ds, hs, ws = np.where(mask > 0)
    if(ds.shape[0]==0):
        return 0, 0, 0, 0, 0, 0
    dmin = max(ds.min() - margin[0], 0)
    dmax = min(ds.max() + margin[0], D)
    hmin = max(hs.min() - margin[1], 0)
    hmax = min(hs.max() + margin[1], H)
    wmin = max(ws.min() - margin[2], 0)
    wmax = min(ws.max() + margin[2], W)
    return dmin, dmax, hmin, hmax, wmin, wmax

def test_no_adapt(net, image):
    image = image.cuda()
    image = image.float()
    with torch.no_grad():
        y1 = net.forward_no_adapt(image)
        y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()[0] 
    
    return label

class Prototype_Pool(nn.Module):
    def __init__(self, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num=class_num
        self.max_length = max
        self.feature_bank = torch.tensor([]).cuda()
        self.image_bank = torch.tensor([]).cuda()
        self.hist_bank = []
        self.mask_bank = torch.tensor([]).cuda()
        self.name_list = []
        
    def get_pool_feature(self, x, mask, top_k = 5):
        if len(self.feature_bank)>0:
            cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2)
            if self.feature_bank.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :self.feature_bank.shape[0]]
                # outall = outall.repeat(1,top_k)

            # rates = cosine_similarities[0][outall[0]].mean(0)
            rates = 1
            weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
            x = x * (1-rates)
            for i in range(min(top_k,self.feature_bank.shape[0])):
                x += self.feature_bank[outall[:,i]]*weight[i]
            return x, len(self.feature_bank)
        else:
            return x, len(self.feature_bank)

    def get_pool_hist(self, hist, top_k = 1):
        if len(self.hist_bank)>0:
            cosine_similarities = torch.nn.functional.cosine_similarity(torch.tensor(hist).unsqueeze(0), torch.tensor(np.array(self.hist_bank)), dim=1)
            if len(self.hist_bank) >= top_k:
                outall = cosine_similarities.argsort(dim=0, descending=True)[:top_k]
            else:
                outall = cosine_similarities.argsort(dim=0, descending=True)[:len(self.hist_bank)]
                # outall = outall.repeat(1,top_k)

            hist = self.hist_bank[outall[0]]
            # rates = cosine_similarities[outall[0]].mean(0)
            # weight = rates * torch.exp(cosine_similarities[outall[0]]) / torch.sum(torch.exp(cosine_similarities[outall[0]]))
            # hist = hist * (1-np.array(rates))
            # for i in range(min(top_k,len(self.hist_bank))):
            #     hist += self.hist_bank[outall[i]]*weight[i]
            return hist
        else:
            return hist
        
    def update_feature_pool(self, feature):
        if self.feature_bank.shape[0] == 0:
            self.feature_bank = torch.cat([self.feature_bank, feature.detach()],dim=0)
        else:
            if self.feature_bank.shape[0] < self.max_length:
                self.feature_bank = torch.cat([self.feature_bank, feature.detach()],dim=0)
            else:
                self.feature_bank = torch.cat([self.feature_bank[-self.max_length:], feature.detach()],dim=0)
                
    def update_image_pool(self, image):
        if self.image_bank.shape[0] == 0:
            self.image_bank = torch.cat([self.image_bank, image.detach()],dim=0)
        else:
            if self.image_bank.shape[0] < self.max_length:
                self.image_bank = torch.cat([self.image_bank, image.detach()],dim=0)
            else:
                self.image_bank = torch.cat([self.image_bank[-self.max_length:], image.detach()],dim=0)
    
    def update_hist_pool(self, hist):
        if len(self.hist_bank) == 0:
            self.hist_bank.append(hist)
        else:
            if len(self.hist_bank) < self.max_length:
                self.hist_bank.append(hist)
            else:
                self.hist_bank.pop(0)
                self.hist_bank.append(hist)
                            
    def update_mask_pool(self, image):
        if self.mask_bank.shape[0] == 0:
            self.mask_bank = torch.cat([self.mask_bank, image.detach()],dim=0)
        else:
            if self.mask_bank.shape[0] < self.max_length:
                self.mask_bank = torch.cat([self.mask_bank, image.detach()],dim=0)
            else:
                self.mask_bank = torch.cat([self.mask_bank[-self.max_length:], image.detach()],dim=0)
                
    def update_name_pool(self, image):
        if len(self.name_list) == 0:
            self.name_list.append(image)
        else:
            if len(self.name_list) < self.max_length:
                self.name_list.append(image)
            else:
                self.name_list = self.name_list[-self.max_length:]
                self.name_list.append(image)
        
class ImgUpdate():
    def __init__(self):
        self.est_ema = None
        self.hist_ema = None
        self.pool = Prototype_Pool(class_num=4, max = 1)
        self.est_list = []
        self.max_len = 1
        self.mean = None
        self.std = None
        
        self.WT_est_ema = None
        self.WT_hists_ema = None
        self.TC_est_ema = None
        self.TC_hists_ema = None
        self.ET_est_ema = None
        self.ET_hists_ema = None
        
    
    def apply_histogram_matching(self, target_image, target_hist, ref_hist, hist_range):
        matched_image = modal_histogram_matching(target_image, target_hist, ref_hist, hist_range)
        return matched_image

    def get_class_label(self, pred, label_list):
        pred_sub = np.zeros_like(pred)
        for lab in label_list:
            pred_sub = pred_sub + np.asarray(pred == lab, np.uint8)
        
        return pred_sub
    
    def img_update_with_class(self, model, input, pred, est_WT, est_TC, est_ET):
        hist_range = (0,1)
        img_hists, bin_edges = calculate_histogram(input, hist_range)
        WT_pred = self.get_class_label(pred, [1,2,3])
        TC_pred = self.get_class_label(pred, [2,3])
        ET_pred = self.get_class_label(pred, [3])
        
        # 不同的类别，分别保存自己效果好的hists
        if self.WT_est_ema == None:
            self.WT_est_ema = est_WT
            self.TC_est_ema = est_TC
            self.ET_est_ema = est_ET
            self.WT_hists_ema = img_hists
            self.TC_hists_ema = img_hists
            self.ET_hists_ema = img_hists
        
        if(est_WT<self.WT_est_ema):
            # 当前WT表现不好，此处得到更新后的WT结果
            updated_WT_input = self.apply_histogram_matching(input, img_hists, self.WT_hists_ema, hist_range)
            updated_WT_pred = test_no_adapt(model, updated_WT_input)
            WT_pred = self.get_class_label(updated_WT_pred, [1,2,3])
        else:
            # 当前WT表现好，用表现好的Hist进行更新
            self.WT_hists_ema = 0.9 * self.WT_hists_ema + 0.1 * img_hists
        
        if(est_TC<self.TC_est_ema):
            # 当前TC表现不好，此处得到更新后的WT结果
            updated_TC_input = self.apply_histogram_matching(input, img_hists, self.TC_hists_ema, hist_range)
            updated_TC_pred = test_no_adapt(model, updated_TC_input)
            TC_pred = self.get_class_label(updated_TC_pred, [2,3])
        else:
            # 当前TC表现好，用表现好的Hist进行更新
            self.TC_hists_ema = 0.9 * self.TC_hists_ema + 0.1 * img_hists
            
        if(est_ET<self.ET_est_ema):
            # 当前ET表现不好，此处得到更新后的WT结果
            updated_ET_input = self.apply_histogram_matching(input, img_hists, self.ET_hists_ema, hist_range)
            updated_ET_pred = test_no_adapt(model, updated_ET_input)
            ET_pred = self.get_class_label(updated_ET_pred, [3])
        else:
            # 当前ET表现好，用表现好的Hist进行更新
            self.ET_hists_ema = 0.9 * self.ET_hists_ema + 0.1 * img_hists
        
        self.WT_est_ema = 0.9 * self.WT_est_ema + 0.1 * est_WT
        self.TC_est_ema = 0.9 * self.TC_est_ema + 0.1 * est_TC
        self.ET_est_ema = 0.9 * self.ET_est_ema + 0.1 * est_ET
        
        final_pred = np.zeros_like(pred)
        final_pred[WT_pred==1] = 1
        final_pred[TC_pred==1] = 2
        final_pred[ET_pred==1] = 3
        
        return final_pred
    
    def img_update_for_tumor(self, input, pred, est_WT, est_TC, est_ET):
        # 只对预测出的肿瘤部分，用bounding box选出，再做后续的处理
        dmin, dmax, hmin, hmax, wmin, wmax = get_3d_crop_bounding_box(mask=pred, margin=[5,10,10])
        tumor = input[:,:,dmin:dmax, hmin:hmax, wmin: wmax]
        # 只计算大脑的hist
        hist_range = (0,1)
        img_hists, bin_edges = calculate_histogram(tumor, hist_range)
        img_est = (est_WT+est_TC+est_ET)/3
        
        # 直接对每个模态进行操作
        if self.est_ema == None:
            self.est_ema = img_est
            self.hists_ema = img_hists
        
        if(img_est<=self.est_ema):
            alpha = 1.0
            beta = 0.0
            # 如果效果不好，则对图片进行更新，同时不更新保存好的直方图
            update_image = True
        else:
            alpha = 0.9
            beta = 0.1
            # 如果效果好呢，则不对图片进行更新，但是要更新保存好的直方图
            update_image = False
        
        updated_hists = alpha * self.hists_ema + beta * img_hists
        # 这里是只用更好的Dice对est进行更新，但是这样存在问题：
        # est只会变得越来越高，同时如果第一个case的est就很高的话，后续的稍微差一点点的case都更新不了直方图，同时也会导致稍微差一点点的case，变得更差
        # updated_est = alpha * self.est_ema + beta * img_est
        
        # 这里对est采取全局更新的策略，能够保证est表征着全局的一个case好坏信息，此时再用这个est_ema去选case，
        # 选的就不是比上一个case，更好的case，而是和到目前为止全局的est相比，表现更好的case
        updated_est = 0.9 * self.est_ema + 0.1 * img_est
        
        updated_input = copy.deepcopy(input)
        updated_tumor = copy.deepcopy(tumor)
        if(update_image):
            updated_tumor = self.apply_histogram_matching(tumor, img_hists, updated_hists, hist_range)
            updated_input[:,:,dmin:dmax, hmin:hmax, wmin: wmax] = updated_tumor
        self.hists_ema = updated_hists
        self.est_ema = updated_est
        
        # plt.figure(figsize=(12, 6))
        # # Plot images
        # plt.subplot(3, 3, 1)
        # plt.imshow(input[0][0][60], cmap='gray')
        # plt.title('Image 1')

        # plt.subplot(3, 3, 3)
        # plt.imshow(updated_input[0][0][60], cmap='gray')
        # plt.title('Image 2')

        # # Plot histograms
        # plt.subplot(3, 3, 4)
        # plt.bar(bin_edges[0][:-1], img_hists[0], width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 1)')

        # plt.subplot(3, 3, 5)
        # plt.bar(bin_edges[0][:-1], self.hists_ema[0], width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 2)')
        
        # updated_img_hists, bin_edges = calculate_histogram(updated_tumor, hist_range)
        # plt.subplot(3, 3, 6)
        # plt.bar(bin_edges[0][:-1], updated_img_hists[0], width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 3)')
        
        # # Plot histograms
        # plt.subplot(3, 3, 7)
        # plt.bar(bin_edges[0][:-1], calculate_cdf(img_hists[0]), width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 1)')

        # plt.subplot(3, 3, 8)
        # plt.bar(bin_edges[0][:-1], calculate_cdf(self.hists_ema[0]), width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 2)')

        # plt.subplot(3, 3, 9)
        # plt.bar(bin_edges[0][:-1], calculate_cdf(updated_img_hists[0]), width=np.diff(bin_edges[0]), edgecolor='black', align='edge')
        # plt.title('Histogram (Image 3)')

        # plt.tight_layout()
        # plt.savefig('/mnt/data1/ZhouFF/TTA4MIS/debug/SSA_hisupdated/1.png')
        
        return updated_input
    
    def img_update(self, input, pred, est_WT, est_TC, est_ET):
        # 针对full image
        # 只计算大脑的hist
        input = input.cpu()
        # 在这里做一下clip
        input[input < -1.0] = -1.0
        input[input > 1.0] = 1.0
        
        updated_input = deepcopy(input)
        hist_range = (-1,1)
        for i in range(input.shape[0]):
            input_sub = input[i]
            img_hist, bin_edge = calculate_histogram(input_sub, hist_range)
            img_est = (est_WT[i]+est_TC[i]+est_ET[i])/3
            
            if self.est_ema == None:
                self.est_ema = img_est
                self.hist_ema = img_hist
            
            if(img_est<self.est_ema):
                alpha = 1.0
                beta = 0.0
                update_image = True
            else:
                alpha = 0.9
                beta = 0.1
                update_image = False
            
            updated_hist = 0.9 * self.hist_ema + 0.1 * img_hist
            updated_est = 0.9 * self.est_ema + 0.1 * img_est
            
            updated_input_sub = copy.deepcopy(input_sub)
            if(update_image):
                updated_input_sub = self.apply_histogram_matching(input_sub, img_hist, updated_hist, hist_range)
            self.hist_ema = updated_hist
            self.est_ema = updated_est
            
            updated_input[i] = updated_input_sub
            
            # plt.figure(figsize=(12, 6))
            # # Plot images
            # plt.subplot(3, 3, 1)
            # plt.imshow(input_sub[0][0], cmap='gray')
            # plt.title('Image 1')

            # plt.subplot(3, 3, 3)
            # plt.imshow(updated_input_sub[0][0], cmap='gray')
            # plt.title('Image 2')

            # # Plot histograms
            # plt.subplot(3, 3, 4)
            # plt.bar(bin_edge[:-1], img_hist, width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 1)')

            # plt.subplot(3, 3, 5)
            # plt.bar(bin_edge[:-1], self.hist_ema, width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 2)')
            
            # updated_img_hist, bin_edge = calculate_histogram(updated_input_sub, hist_range)
            # plt.subplot(3, 3, 6)
            # plt.bar(bin_edge[:-1], updated_img_hist, width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 3)')
            
            # # Plot histograms
            # plt.subplot(3, 3, 7)
            # plt.bar(bin_edge[:-1], calculate_cdf(img_hist), width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 1)')

            # plt.subplot(3, 3, 8)
            # plt.bar(bin_edge[:-1], calculate_cdf(self.hist_ema), width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 2)')

            # plt.subplot(3, 3, 9)
            # plt.bar(bin_edge[:-1], calculate_cdf(updated_img_hist), width=np.diff(bin_edge), edgecolor='black', align='edge')
            # plt.title('Histogram (Image 3)')

            # plt.tight_layout()
            # plt.savefig('/mnt/data1/ZhouFF/TTA4MIS/debug/SSA_hisupdated/1.png')
        
        return updated_input
    
    def img_update_histbank(self, input, pred, est_WT, est_TC, est_ET, est_avg):
        # 针对full image
        # 只计算大脑的hist
        input = input.cpu()
        # 在这里做一下clip
        input[input < -1.0] = -1.0
        input[input > 1.0] = 1.0
        
        updated_input = deepcopy(input)
        hist_range = (-1,1)
        
        self.est_list.extend(est_avg.tolist())
        estp80 = np.percentile(self.est_list, 80)
        print(f'estp80:{estp80}')
        
        for i in range(input.shape[0]):
            input_sub = input[i]
            img_hist, bin_edge = calculate_histogram(input_sub, hist_range)
            img_est = est_avg[i]
            
            # if self.est_ema == None:
            #     self.est_ema = img_est
                # self.hist_ema = img_hist
            
            if(img_est<estp80):
                alpha = 1.0
                beta = 0.0
                update_image = True
            else:
                alpha = 0.9
                beta = 0.1
                update_image = False
                self.pool.update_hist_pool(img_hist)
            
            # updated_est = 0.9 * self.est_ema + 0.1 * img_est
            
            updated_input_sub = copy.deepcopy(input_sub)
            if(update_image):
                updated_hist = self.pool.get_pool_hist(img_hist)
                updated_input_sub = self.apply_histogram_matching(input_sub, img_hist, updated_hist, hist_range)
            # self.est_ema = updated_est
            
            updated_input[i] = updated_input_sub
            
            # if(update_image):
            #     plt.figure(figsize=(12, 6))
            #     # Plot images
            #     plt.subplot(3, 3, 1)
            #     plt.imshow(input_sub[0][0], cmap='gray')
            #     plt.title('Image 1')

            #     plt.subplot(3, 3, 3)
            #     plt.imshow(updated_input_sub[0][0], cmap='gray')
            #     plt.title('Image 2')

            #     # Plot histograms
            #     plt.subplot(3, 3, 4)
            #     plt.bar(bin_edge[:-1], img_hist, width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 1)')

            #     plt.subplot(3, 3, 5)
            #     plt.bar(bin_edge[:-1], updated_hist, width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 2)')
                
            #     updated_img_hist, bin_edge = calculate_histogram(updated_input_sub, hist_range)
            #     plt.subplot(3, 3, 6)
            #     plt.bar(bin_edge[:-1], updated_img_hist, width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 3)')
                
            #     # Plot histograms
            #     plt.subplot(3, 3, 7)
            #     plt.bar(bin_edge[:-1], calculate_cdf(img_hist), width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 1)')

            #     plt.subplot(3, 3, 8)
            #     plt.bar(bin_edge[:-1], calculate_cdf(updated_hist), width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 2)')

            #     plt.subplot(3, 3, 9)
            #     plt.bar(bin_edge[:-1], calculate_cdf(updated_img_hist), width=np.diff(bin_edge), edgecolor='black', align='edge')
            #     plt.title('Histogram (Image 3)')

            #     plt.tight_layout()
            #     plt.savefig('/mnt/data1/ZhouFF/TTA4MIS/debug/SSA_hisupdated/1.png')
        
        return updated_input
    
    def img_update_meanstd(self, input, pred, est_WT, est_TC, est_ET):
            # 针对full image
            # 只计算大脑的hist
            input = input.cpu()
            updated_input = deepcopy(input)
            for i in range(input.shape[0]):
                input_sub = input[i]
                
                img_mean = input_sub.mean()
                img_std = input_sub.std()
                img_est = (est_WT[i]+est_TC[i]+est_ET[i])/3
                
                if self.est_ema == None:
                    self.est_ema = img_est
                    self.mean = img_mean
                    self.std = img_std
                
                if(img_est<=self.est_ema):
                    alpha = 1.0
                    beta = 0.0
                    update_image = True
                else:
                    alpha = 0.9
                    beta = 0.1
                    update_image = False
                
                updated_mean = alpha * self.mean + beta * img_mean
                updated_std = alpha * self.std + beta * img_std
                updated_est = 0.9 * self.est_ema + 0.1 * img_est
                
                updated_input_sub = copy.deepcopy(input_sub)
                if(update_image):
                    updated_input_sub = input_sub * updated_std + updated_mean
                self.mean = updated_mean
                self.std = updated_std
                self.est_ema = updated_est
                
                updated_input[i] = updated_input_sub
            
            return updated_input
        
    def img_update_featbank(self, features, pred, est_WT, est_TC, est_ET, est_avg):
        self.est_list.extend(est_avg.tolist())
        estp90 = np.percentile(self.est_list, 95)
        print(f'estp90:{estp90}')
        
        updated_feat = torch.zeros_like(features)
        for i in range(features.shape[0]):
            latent_model = features[i].unsqueeze(0)
            
            b,c,w,h = latent_model.shape
            sup_pixel = w
            latent_model = latent_model.reshape(b,c,int(w/sup_pixel),sup_pixel,int(h/sup_pixel),sup_pixel)
            latent_model = latent_model.permute(0,2,4,1,3,5)
            latent_model = latent_model[0].reshape(1*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)
            latent_model_, len_pool = self.pool.get_pool_feature(latent_model,None,top_k = self.max_len)
            
            if(estp90<est_avg[i]):
                # print("feature update by {}, with est_avg {}".format(i, est_avg[i]))
                self.pool.update_feature_pool(latent_model)
                
            latent_model_ = latent_model_.view(b,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
            latent_model_ = latent_model_.permute(0,3,1,4,2,5)
            latent_model_ = latent_model_.reshape(b,c,w,h)

            updated_feat[i] = (est_avg[i]/100) * features[i] + (1-est_avg[i]/100)*latent_model_[0]
            # updated_feat[i] = latent_model_[0]
            
        return updated_feat

        
            