from concurrent.futures import thread
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
import SimpleITK as sitk
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
from utils.utils import rotate_single_random,derotate_single_random,add_gaussian_noise_3d
from robustbench.losses import WeightedCrossEntropyLoss,DiceCeLoss,DiceLoss,center_alignment_loss,KDLoss,mmd_loss
import torch
from skimage import exposure, io, img_as_ubyte
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import os
dicece_loss = DiceCeLoss(4)
# dicece_loss = DiceCELoss(lambda_dice=0)
class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model,anchor,anchor_model,ema_model):
        super().__init__()
        self.model = model
        self.model_anchor = anchor_model.eval()
        # self.optimizer = optimizer
        self.param_model_list = []
        self.param_prompt_list = []

    def forward(self, x, label_batch, names):
        self.label = label_batch
        
        if self.episodic:
            self.reset()
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, names)
        return outputs

    def forward_and_adapt(self, x, model, names):
        layer_fea = 'med'
        self.ww = x.shape[-1]
        bad_num = x.shape[0]
        topk = 5
        
        latent_model = self.model_anchor.get_feature(x, loc = layer_fea)
        # b,c,w,h = latent_model.shape
        # sup_pixel = w
        # latent_model = latent_model.reshape(b,c,int(w/sup_pixel),sup_pixel,int(h/sup_pixel),sup_pixel)
        # latent_model = latent_model.permute(0,2,4,1,3,5)
        # latent_model = latent_model[0].reshape(1*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)
        # latent_model = latent_model[0:bad_num].reshape(bad_num*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)
        latent_model_,fff, out_image, out_mask, len_pool = self.pool.get_pool_feature(latent_model,None,top_k = topk)
        with torch.no_grad():
            if len_pool < topk:
                threshold = len_pool / topk
            else:
                threshold = 0.9
            # fine = self.get_fine_hh320(x, self.model_anchor, self.entropy_list, threshold = threshold)
            # fine = self.get_fine_cc(x, self.model_anchor, self.entropy_list, threshold = threshold)
            fine = self.get_fine_cc500(x, self.model_anchor, self.entropy_list, threshold = 0.9)
            # fine = self.get_bad_cc500(x, self.model_anchor, self.entropy_list, threshold = 0.1)

            # fine = self.get_fine_en(x, self.model_anchor, self.entropy_list, threshold = 0.9)
            # print(fine)
        if fine:
            print(names)
            return self.model_anchor(x)
            # return model(x)
        else:
            return None
        # return self.model_anchor(x)
            # self.pool.update_feature_pool(latent_model)
            # self.pool.update_image_pool(x)
            # self.pool.update_mask_pool(model(x).softmax(1))
            # self.pool.update_name_pool(names[0])
        # name = names[0].split('/')[-1]
        # if out_image is not None:
        #     out_image = out_image[0]
        #     x_hised = torch.cat((x, out_image), dim=0)
        #     latent_model = model.get_feature(x_hised, loc = layer_fea)
        #     latent_model_ = latent_model_.view(bad_num,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
        #     latent_model_ = latent_model_.permute(0,3,1,4,2,5)
        #     latent_model_ = latent_model_.reshape(bad_num,c,w,h)
        #     latent_model[0:1] = latent_model_
        #     outputs2 = model.get_output(latent_model,loc = layer_fea)[0:1].softmax(1)
        #     return outputs2, latent_model_.cpu().numpy()
        # else:
        #     # return model(x)
        #     return self.model_anchor(x), None

            # x_hised = self.get_his_image(x,out_image)
            #     out_image_np = out_image[0][0].cpu().numpy()
            #     predict_dir  = os.path.join('results-mms2d/hised2/','ref',name)
            #     out_lab_obj = sitk.GetImageFromArray(out_image_np/1.0)
            #     sitk.WriteImage(out_lab_obj, predict_dir)
            #     out_image_np = x_hised[0][0].cpu().numpy()
            #     predict_dir  = os.path.join('results-mms2d/hised2/','out',name)
            #     out_lab_obj = sitk.GetImageFromArray(out_image_np/1.0)
            #     sitk.WriteImage(out_lab_obj, predict_dir)
            #     out_image_np = x[0][0].cpu().numpy()
            #     predict_dir  = os.path.join('results-mms2d/hised2/','source',name)
            #     out_lab_obj = sitk.GetImageFromArray(out_image_np/1.0)
            #     sitk.WriteImage(out_lab_obj, predict_dir)
        

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def get_his_image(self,source_image,target_image):
        # print(source_image.shape,target_image.shape,'91',source_image.max(),source_image.min(),target_image.max(),target_image.min())
        source_image = source_image[0][0].cpu().numpy()
        target_image = target_image[0][0].cpu().numpy()
        source_image[source_image>1] = 1.0
        target_image[target_image>1] = 1.0
        source_image[source_image<-1] = -1.0
        target_image[target_image<-1] = -1.0


        image1 = (source_image + 1.) / 2.
        image2 = (target_image + 1.) / 2.

        matched_image = exposure.match_histograms(image1, image2)  
        tensor = torch.from_numpy(matched_image*2.0 - 1.0).cuda()
        tensor = tensor.unsqueeze(0).unsqueeze(0).float()
        return tensor

    @torch.no_grad()  
    def get_SND(self, x, model_anchor):
        with torch.no_grad():
            b,c,w,h = x.shape
            entropy_list = []
            for i in range(b):
                with torch.no_grad():
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = anchor_prd.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                # print(pred1.shape,'183') ## torch.Size([102400, 4])
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1) )
                entropy_list.append(pred1_en)
        return entropy_list  
    ## c*c 
    def get_fine_cc(self, x, model_anchor,entropy_list, threshold = 0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = anchor_prd.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1 = F.normalize(pred1)
                # print(pred1.shape,'183') ## torch.Size([102400, 4])
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1) )
                entropy_list.append(pred1_en)
                entropy_list = sorted(entropy_list)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        ten_percent_index = int(len(entropy_list) * (1 - threshold))
        if ten_percent_index>0:
            ten_percent_min_value = entropy_list[:ten_percent_index][-1]
            return pred1_en <= ten_percent_min_value
        else:
            return False

    def get_fine_hh320(self, x, model_anchor,entropy_list, threshold = 0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 320
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en =  self.entropy(torch.matmul(pred1, pred1.t()) * 20)
                entropy_list.append(pred1_en)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            return pred1_en <= ten_percent_min_value
        else:
            return False

    ## entropy 0.9
    def get_fine_en(self, x, model_anchor,entropy_list,threshold = 0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1_en =  self.entropy(anchor_prd, prob=False)
                entropy_list.append(pred1_en)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            # print(pred1_en,'123',ten_percent_min_value,ten_percent_index,sorted_list[:ten_percent_index][-1],sorted_list[:ten_percent_index][0])
            return pred1_en <= ten_percent_min_value
        else:
            return False

    def get_fine_cc500(self, x, model_anchor,entropy_list, threshold = 0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                # select_point = 500
                # pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                select_point = 500
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                # pred1_en =  self.entropy(torch.matmul(pred1, pred1.t()) * 20)
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1))
                # pred1_en =  self.entropy(pred1) 
                entropy_list.append(pred1_en)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        # print(ten_percent_index, len(sorted_list), self.max_lens, threshold )
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            # print(pred1_en,'123',ten_percent_min_value,ten_percent_index,sorted_list[:ten_percent_index][-1],sorted_list[:ten_percent_index][0])

            return pred1_en <= ten_percent_min_value
        else:
            return False
    def get_bad_cc500(self, x, model_anchor,entropy_list, threshold = 0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 500
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1))
                entropy_list.append(pred1_en)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        # print(ten_percent_index, len(sorted_list), self.max_lens, threshold )
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            # print(pred1_en,'123',ten_percent_min_value,ten_percent_index,sorted_list[:ten_percent_index][-1],sorted_list[:ten_percent_index][0])

            return pred1_en >= ten_percent_min_value
        else:
            return False
        
  

        
    
    def get_dense_sup(self, x, model_anchor, norm_model):
        b,c,w,h = x.shape
        entropy_list = []
        mean_list = []
        var_list = []
        for i in range(b):
            anchor_prd = model_anchor(x[i:i+1]).softmax(1)
            if anchor_prd[0][1:].sum() < 1000:
                entropy = 1
            else:
                entropy = -(anchor_prd * torch.log2(anchor_prd + 1e-10)).sum() / (w*h)
            entropy_list.append(entropy)
            norm_model(x[i:i+1])
            layer_last = norm_model.enc.down_path[4]   
            target_bn_layer = layer_last.conv_conv[1]
            if target_bn_layer is not None:
                batch_mean = target_bn_layer.running_mean.clone().detach()
                batch_var = target_bn_layer.running_var.clone().detach()
                mean_list.append(batch_mean.mean())
                var_list.append(batch_var.mean())
            else:
                print("Target BN layer not found.")
        return entropy_list, mean_list, var_list
    
    def find_closest_and_min(self, c_list):
        best_score = float('inf')
        best_index = None
        for i, c_value in enumerate(zip(c_list)):
            total_score = c_value
            if total_score[0] < best_score:
                best_score = total_score[0]
                best_index = i
        return best_index
    
    def find_closest_and_max(self, c_list):
        best_score = float('-inf')
        best_index = None
        for i, c_value in enumerate(zip(c_list)):
            total_score = c_value
            if total_score[0] > best_score:
                best_score = total_score[0]
                best_index = i
        return best_index
    def find_sorted_indices(self, c_list):
        # 创建一个包含 (索引, 值) 元组的列表，并按值从大到小排序
        sorted_indices = sorted(enumerate(c_list), key=lambda x: x[1], reverse=True)
        # 提取排序后的索引
        sorted_indices = [index for index, _ in sorted_indices]
        
        return sorted_indices

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def randomHorizontalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        horizontal_flip = torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * horizontal_flip
        x = apply_affine(x, affine)
        return x, affine.detach()

    def randomRotate(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        rotation = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = rotation.repeat(mask.sum(), 1, 1)

        x = apply_affine(x, affine)
        
        return x, affine.detach()
    def randomVerticalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        vertical_flip = torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * vertical_flip
        x = apply_affine(x, affine)
        return x, affine.detach()
    def randomResizeCrop(self, x):
        # TODO: Investigate different scale for x and y
        delta_scale_x = 0.2
        delta_scale_y = 0.2

        scale_matrix_x = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        scale_matrix_y = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        translation_matrix_x = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        translation_matrix_y = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        delta_x = 0.5 * delta_scale_x * (2*torch.rand(x.size(0), 1, 1, device=x.device) - 1.0)
        delta_y = 0.5 * delta_scale_y * (2*torch.rand(x.size(0), 1, 1, device=x.device) -1.0)

        random_affine = (1 - delta_scale_x) * scale_matrix_x + (1 - delta_scale_y) * scale_matrix_y +\
                    delta_x * translation_matrix_x + \
                    delta_y * translation_matrix_y

        x = apply_affine(x, random_affine)
        return x, random_affine.detach()
    def get_pseudo_label(self, net, x, mult=3):
        preditions_augs = []
        # if is_training:
        #     net.eval()
        outnet = net(x)
        # preditions_augs.append(F.softmax(outnet, dim=1))
        preditions_augs.append(outnet)

        for i in range(mult-1):
            x_aug, rotate_affine = self.randomRotate(x)
            x_aug, vflip_affine = self.randomVerticalFlip(x_aug)
            x_aug, hflip_affine = self.randomHorizontalFlip(x_aug)

            # x_aug, hflip_affine = self.randomHorizontalFlip(x)
            # x_aug, crop_affine  = self.randomResizeCrop(x_aug)

            # get label on x_aug
            outnet = net(x_aug)
            pred_aug = outnet
            
            pred_aug = F.softmax(pred_aug, dim=1)
            pred_aug = apply_invert_affine(pred_aug, rotate_affine)
            pred_aug = apply_invert_affine(pred_aug, vflip_affine)
            pred_aug = apply_invert_affine(pred_aug, hflip_affine)

            preditions_augs.append(pred_aug)


        preditions = torch.stack(preditions_augs, dim=0).mean(dim=0) # batch x n_classes x h x w
        # renormalize the probability (due to interpolation of zeros, mean does not imply probability distribution over the classes)
        preditions = preditions / torch.sum(preditions, dim=1, keepdim=True)
        # if is_training:
        #     net.train()
        return preditions



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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    # optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, ema_model


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_tent_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def configure_norm_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
    
def update_ema_variables(ema_model, model, alpha_teacher):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    for ema_param, param in zip(ema_model.enc.parameters(), model.enc.parameters()):
    # for ema_param, param in zip(ema_model.dec1.parameters(), model.dec1.parameters()):
        # ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def data_augmentation(image):
    random_numbers = random.sample(range(4), 2)
    if 0 in random_numbers:
        image = image.flip(-1)           # 水平翻转
    if 1 in random_numbers:
        image = image.flip(-2)             # 垂直翻转
    if 2 in random_numbers:
        image = FF.rotate(image, 90)
    if 3 in random_numbers:
        image = FF.adjust_brightness(image, brightness_factor=0.2)              # 调整亮度
    return image
import numpy as np
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) 
def configure_debn_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(True)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(False)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
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

def apply_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.size(), align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return out

def configure_model(model, eps, momentum, reset_stats, no_stats):
    model.eval()
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()

            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

class ClassRatioLoss__(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ClassRatioLoss, self).__init__()
        self.temperature = temperature

    def forward(self, predicted_probs1, predicted_probs2):
        # Ensure input shapes are as expected
        assert predicted_probs1.shape == predicted_probs2.shape
        b, c, w, h = predicted_probs1.shape

        # Apply softmax along the channel dimension to get class probabilities
        probs1 = F.softmax(predicted_probs1, dim=1)
        probs2 = F.softmax(predicted_probs2, dim=1)

        # Calculate the class ratios
        class_ratios1 = torch.mean(probs1, dim=(2, 3), keepdim=True)
        class_ratios2 = torch.mean(probs2, dim=(2, 3), keepdim=True)

        # Calculate the KL Divergence between class ratios
        kl_divergence = F.kl_div(
            F.log_softmax(class_ratios1 / self.temperature, dim=1),
            class_ratios2 / self.temperature, reduction='batchmean'
        )
        # print(kl_divergence,'668')
        return kl_divergence

class ClassRatioLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ClassRatioLoss, self).__init__()
        self.temperature = temperature

    def forward(self, predicted_probs1, predicted_probs2):
        # Ensure input shapes are as expected
        assert predicted_probs1.shape == predicted_probs2.shape
        b, c, w, h = predicted_probs1.shape

        # Apply softmax along the channel dimension to get class probabilities
        probs1 = F.softmax(predicted_probs1, dim=1)
        probs2 = F.softmax(predicted_probs2, dim=1)

        # Calculate the class ratios using both methods
        class_ratios1_mean = torch.mean(probs1, dim=(2, 3), keepdim=True)
        class_ratios2_mean = torch.mean(probs2, dim=(2, 3), keepdim=True)

        # Calculate the class ratios by counting pixel numbers
        class_ratios1_pixel = torch.sum(probs1, dim=(2, 3), keepdim=True) / (w * h)
        class_ratios2_pixel = torch.sum(probs2, dim=(2, 3), keepdim=True) / (w * h)

        # Calculate the KL Divergence between class ratios
        kl_divergence1 = F.kl_div(F.log_softmax(class_ratios1_mean / self.temperature, dim=1),
                                  class_ratios2_mean / self.temperature, reduction='batchmean')

        kl_divergence2 = F.kl_div(F.log_softmax(class_ratios1_pixel / self.temperature, dim=1),
                                  class_ratios2_pixel / self.temperature, reduction='batchmean')

        # Combine the losses
        class_ratio_loss = (kl_divergence1 + kl_divergence2) / 2.0

        return class_ratio_loss




def apply_invert_affine(x, affine):
    # affine shape should be batch x 2 x 3
    # x shape should be batch x ch x h x w

    # get homomorphic transform
    H = torch.nn.functional.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0

    inv_H = torch.inverse(H)
    inv_affine = inv_H[:, :2, :3]

    grid = torch.nn.functional.affine_grid(inv_affine, x.size(), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return x
def dice_coefficient(tensor1, tensor2, epsilon=1e-6):
    # Flattening the C, W, H dimensions
    flat_tensor1 = tensor1.view(tensor1.size(0), -1)
    flat_tensor2 = tensor2.view(tensor2.size(0), -1)
    
    # Calculating intersection and union
    intersection = (flat_tensor1 * flat_tensor2).sum(1)
    union = flat_tensor1.sum(1) + flat_tensor2.sum(1)
    
    # Calculating Dice
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice
class Prototype_Pool(nn.Module):
    def __init__(self, delta=0.1, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num=class_num
        self.max_length = max
        self.feature_bank = torch.tensor([]).cuda()
        self.image_bank = torch.tensor([]).cuda()
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

            rates = cosine_similarities[0][outall[0]].mean(0)
            weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
            x = x * (1-rates)
            for i in range(min(top_k,self.feature_bank.shape[0])):
                x += self.feature_bank[outall[:,i]]*weight[i]
            return x,self.feature_bank[outall[:,]],self.image_bank[outall[:,]],self.mask_bank[outall[:,]], len(self.feature_bank)
        else:
            return x,x,None,None, len(self.feature_bank)
        
    # def get_pool_feature(self, xs, mask, top_k = 5):
    #     if len(self.feature_bank)>0:
    #         for i in range(xs.shape[0]):
    #             x = xs[i:i+1]
    #             cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2)
    #             if self.feature_bank.shape[0] >= top_k:
    #                 outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
    #             else:
    #                 outall = cosine_similarities.argsort(dim=1, descending=True)[:, :self.feature_bank.shape[0]]
    #             # outall = outall.repeat(1,top_k)

    #             rates = cosine_similarities[0][outall[0]].mean(0)
    #             weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
    #             x = x * (1-rates)
    #             for i in range(min(top_k,self.feature_bank.shape[0])):
    #                 x += self.feature_bank[outall[:,i]]*weight[i]
    #             xs[i:i+1] = x
    #         return xs,self.feature_bank[outall[:,]],self.image_bank[outall[:,]],self.mask_bank[outall[:,]], len(self.feature_bank)
    #     else:
    #         return xs,xs,None,None, len(self.feature_bank)

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
