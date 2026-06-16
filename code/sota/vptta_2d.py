import os
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, sys, datetime
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=320):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 1, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_

class VPTTA(nn.Module):
    def __init__(self, model, optimizer, prompt):
        super().__init__()
        # Model
        self.model = model
        # Optimizer
        self.optimizer = optimizer
        # Prompt
        self.prompt = prompt
        self.iters = 1
        # Memory Bank
        self.neighbor = 16
        self.memory_bank = Memory(size=40, dimension=self.prompt.data_prompt.numel())
        self.print_prompt()
        print('***' * 20)


    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        
        x= Variable(x)
        self.model.eval()
        self.prompt.train()
        self.model.change_BN_status(new_sample=True)
        
        # Initialize Prompt
        if len(self.memory_bank.memory.keys()) >= self.neighbor:
            _, low_freq = self.prompt(x)
            init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
        else:
            init_data = torch.ones((1, 1, self.prompt.prompt_size, self.prompt.prompt_size)).data
        self.prompt.update(init_data)
        
        for tr_iter in range(self.iters):
            prompt_x, _ = self.prompt(x)
            self.model(prompt_x)
            times, bn_loss = 0, 0
            for nm, m in self.model.named_modules():
                if isinstance(m, AdaBN):
                    bn_loss += m.bn_loss
                    times += 1
            loss = bn_loss / times
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.change_BN_status(new_sample=False)
            
        # Inference
        self.model.eval()
        self.prompt.eval()
        with torch.no_grad():
            prompt_x, low_freq = self.prompt(x)
            output = self.model(prompt_x)
        # Update the Memory Bank
        self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())
        
        return output
         