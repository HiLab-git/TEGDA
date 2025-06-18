from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import random
import torchio as tio


def get_tta_transforms(gaussian_std: float=0.005, soft=False):
    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5
    tta_transforms = []
    tta_transforms.append(tio.RescaleIntensity(out_min_max=(clip_min, clip_max)))
    tta_transforms.append(tio.RandomGamma(log_gamma=(-0.3, 0.3)))
    tta_transforms.append(tio.RandomAffine(
        scales=(0.95, 1.05) if soft else (0.9, 1.1),
        degrees=(-8, 8) if soft else (-15, 15),
        translation=(0.0625, 0.0625, 0.0625),  # Convert to 3D translations
        isotropic=False
    ))
    tta_transforms.append(tio.RandomFlip(axes=(1), flip_probability=p_hflip))
    if soft:
        tta_transforms.append(tio.RandomBlur(std=(0.001, 0.25)))
    else:
        tta_transforms.append(tio.RandomBlur(std=(0.001, 0.5)))
    tta_transforms.append(tio.RandomNoise(mean=0, std=gaussian_std))
    # 根据CoTTA的源代码，好像在最后又进行了一次归一
    tta_transforms.append(tio.RescaleIntensity(out_min_max=(clip_min, clip_max)))
    transform = tio.Compose(tta_transforms)
    
    return transform


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # print('Update the ema_prompt!')
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class VisionPrompt(nn.Module):
    def __init__(self, size):
        super(VisionPrompt, self).__init__()
        # 生成一个Domain-specific Prompt和一个Domain-agnostic Prompt
        self.size = size
        self.dsp = nn.Parameter(torch.zeros(size, size, size),requires_grad=True)
        self.dap = nn.Parameter(torch.zeros(size, size, size),requires_grad=True)
        
    
    def generate_pos(self, d, h, w):
        return (
            random.randint(0,d-self.size),
            random.randint(0,h-self.size),
            random.randint(0,w-self.size),
        )
    
    def rescale(self,x):
        min_val = x.min()
        max_val = x.max()
        # 线性缩放公式
        if min_val == max_val:
        # 返回全零张量或其他适合的值
            return torch.zeros_like(x)
        scaled_x = 2 * (x - min_val) / (max_val - min_val) - 1
        return scaled_x

    def forward(self, x):
        #!!!! 这里有个疑问，是否要对所有通道都进行相同的Prompt？ 感觉通道上，是有idea可以出的？
        b, c, d, h, w = x.shape
        dsp_pos = self.generate_pos(d,h,w)
        # 对Prompt进行尺度变化，让其变化能够在图像上表现更加明显
        # scaled_dsp = self.rescale(self.dsp)
        scaled_dsp = self.dsp
        # dap_pos = self.generate_pos(d,h,w)
        # print(self.dsp)
        # 添加DSP
        x[:,:,dsp_pos[0]:dsp_pos[0]+self.size,dsp_pos[1]:dsp_pos[1]+self.size,dsp_pos[2]:dsp_pos[2]+self.size] += scaled_dsp
        # 添加DAP
        # prompted_x = prompted_x[dap_pos[0]:dap_pos[0]+self.size,dap_pos[1]:dap_pos[1]+self.size,dap_pos[2]:dap_pos[2]+self.size]+self.dap
        # print(prompted_x.shape)
        return x

class VDPTTA(nn.Module):
    """VDPTTA adapts visual prompts for Continual Test-time adaptation
    """
    def __init__(self, prompt, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.prompt = prompt
        self.ema_prompt = deepcopy(prompt)
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs
    
    def forward_dropout(self, x):
        outputs_ema = self.model(self.ema_prompt(x))

        return outputs_ema            

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        x_trans = self.transform(x[0].cpu()).cuda()
        x_trans = torch.unsqueeze(x_trans, 0)
        outputs = self.model(self.prompt(x_trans))
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model(self.ema_prompt(x)), dim=1).max(1)[0]
        # standard_ema = self.model(self.ema_prompt(x))
        # Augmentation-averaged Prediction
        # N = 32 
        # outputs_emas = []
        # for i in range(N):
        #     x_trans = self.transform(x[0].cpu()).cuda()
        #     x_trans = torch.unsqueeze(x_trans, 0)
        #     outputs_  = self.model(self.ema_prompt(x)).detach()
        #     outputs_emas.append(outputs_)
            # print(outputs_.shape,x.shape,self.transform(x).shape,'106')
        # Threshold choice discussed in supplementary
        # if anchor_prob.mean()<self.ap:
        #     outputs_ema = torch.stack(outputs_emas).mean(0)
        # else:
        #     outputs_ema = standard_ema
        # Teacher Prediction
        outputs_ema = self.model(self.ema_prompt(x))
        # Student update
        loss = ((softmax_entropy(outputs, outputs_ema)+softmax_entropy(outputs_ema, outputs))/2).mean(0) 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        self.ema_prompt = update_ema_variables(ema_model = self.ema_prompt, model = self.prompt, alpha_teacher=self.mt)
        
        # 使用更新后的prompt重新输出一次结果
        # outputs_ema = self.model(self.ema_prompt(x))

        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    b, c, d, h, w =  x.shape
    # print(x.shape,x_ema.shape,'145')
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (b * d * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    # print(entropy1)
    return entropy1

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = [model.dsp,model.dap]
    names = ['dsp','dap']
    
    return params, names


def configure_prompt(size):
    """Configure model for use with tent."""
    prompt = VisionPrompt(size=size)
    # print(prompt.__dict__)
    prompt = prompt.cuda()
    prompt.train()
    return prompt

