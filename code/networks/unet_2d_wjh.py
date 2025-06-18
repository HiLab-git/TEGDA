# from loss import DiceLoss, DiceCeLoss
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler
from torch.nn import init

def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>




class UNetConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(UNetConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)
 
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, dropout_p):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTransposed2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode=='upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, dropout_p)

    def centre_crop(self, layer, target_size):
        _,_,layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.centre_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out

class Encoder(nn.Module):
    def __init__(self,
        in_chns,
        n_classes,
        ft_chns,
        dropout_p
        ):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.down_path = nn.ModuleList()
        self.down_path.append(UNetConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))

    def forward(self, x):
        blocks=[]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x ,2)
        return blocks, x

class aux_Decoder(nn.Module):
    def __init__(self, 
        in_chns,
        n_classes,
        ft_chns,
        dropout_p,
        up_mode):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(self.ft_chns[4], self.ft_chns[3], up_mode, self.dropout[1]))
        self.up_path.append(UNetUpBlock(self.ft_chns[3], self.ft_chns[2], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[2], self.ft_chns[1], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[1], self.ft_chns[0], up_mode, self.dropout[0]))
        self.last = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i -1])
        return self.last(x)
class Decoder__(nn.Module):
    """Decoder part of U-Net"""
    def __init__(self, ft_chns, dropout_p, up_mode):
        """
        Args:
            ft_chns (list of int): Feature channels for each block.
            dropout_p (list of float): Dropout probabilities for each block.
            up_mode (str): Upsampling mode - 'upconv' or 'upsample'.
        """
        super().__init__()
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(ft_chns[4], ft_chns[3], up_mode, dropout_p[4]))
        self.up_path.append(UNetUpBlock(ft_chns[3], ft_chns[2], up_mode, dropout_p[3]))
        self.up_path.append(UNetUpBlock(ft_chns[2], ft_chns[1], up_mode, dropout_p[2]))
        self.up_path.append(UNetUpBlock(ft_chns[1], ft_chns[0], up_mode, dropout_p[1]))
        self.last = nn.Conv2d(ft_chns[0], 4, kernel_size=1)  # Output channel set to 1

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        x = self.last(x)
        return x

class Decoder(nn.Module):
    """Decoder part of U-Net"""
    def __init__(self, ft_chns, n_class, dropout_p, up_mode):
        """
        Args:
            ft_chns (list of int): Feature channels for each block.
            dropout_p (list of float): Dropout probabilities for each block.
            up_mode (str): Upsampling mode - 'upconv' or 'upsample'.
        """
        super().__init__()
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(ft_chns[4], ft_chns[3], up_mode, dropout_p[4]))
        self.up_path.append(UNetUpBlock(ft_chns[3], ft_chns[2], up_mode, dropout_p[3]))
        self.up_path.append(UNetUpBlock(ft_chns[2], ft_chns[1], up_mode, dropout_p[2]))
        self.up_path.append(UNetUpBlock(ft_chns[1], ft_chns[0], up_mode, dropout_p[1]))
        self.last = nn.Conv2d(ft_chns[0], n_class, kernel_size=1)  # Output channel set to 1

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        x = self.last(x)
        return x

class UNet(nn.Module):
    def __init__(self,in_chnss,n_classess,Tanh_gene):
        super(UNet, self).__init__()
        lr = 0.001
        in_chns = in_chnss
        n_classes = n_classess
        ft_chns = [32, 64, 128, 256, 512]
        dropout_p = [0,0,0,0,0]
        up_mode = 'upsample'
        self.Tanh_gene_bool = Tanh_gene
        
        self.encoder = Encoder(in_chns,n_classes,ft_chns,dropout_p)
        self.decoder = Decoder(ft_chns,n_classes,dropout_p,up_mode)
        self.Tanh_gene = nn.Tanh()
        opt = 'adam'
        if opt == 'adam':
            # Combine the parameters of both encoder and decoder
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                lr=lr, 
                betas=(0.9, 0.999)
            )
        elif opt == 'SGD':
            self.optimizer = torch.optim.SGD(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                lr=lr, 
                momentum=0.9
            )
        # self.criterion = nn.MSELoss()  
        self.criterion = nn.L1Loss()
        self.optimizer_sch = get_scheduler(self.optimizer)

    def initialize(self):
        init_weights(self.enc)
        init_weights(self.aux_dec1)

    def update_lr(self):
        self.optimizer_sch.step()

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        blocks, bottleneck = self.encoder(x)
        output = self.decoder(bottleneck, blocks)
        if self.Tanh_gene_bool:
            output = self.Tanh_gene(output)
        return output

    def train_source_1(self, imagesa, labelsa):
        self.imgA = imagesa
        self.labA = labelsa

        # 修正拼写错误
        self.optimizer.zero_grad()  # 清除上一次的梯度
        output = self.forward(self.imgA)  # 前向传播
        seg_loss_B = self.criterion(output, self.labA)  # 计算损失
        seg_loss_B.backward()  # 反向传播，计算梯度
        self.optimizer.step()  # 更新模型参数

        self.loss_seg = seg_loss_B.item()  # 记录损失
        print('train_source_1', self.loss_seg)
        return output

    def fine_tune(self, imagesa, labelsa):
        self.imgA = imagesa
        self.labA = labelsa

        self.optimizer.zero_grad()  
        output = self.forward(self.imgA)  
        seg_loss_B = self.segloss(output, self.labA, one_hot = True)
        seg_loss_B.backward()  # 反向传播，计算梯度
        self.optimizer.step()  # 更新模型参数

        self.loss_seg = seg_loss_B.item()  # 记录损失
        print('train_source_1', self.loss_seg)
        return output
    def test_1(self, imagesa):
        output = self.forward(imagesa)  
        return output