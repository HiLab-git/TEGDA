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
import cv2
from utils.third_party import aug
from utils.prepare_dataset import te_transforms
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
from utils.utils import rotate_single_random,derotate_single_random,add_gaussian_noise_3d
from robustbench.losses import WeightedCrossEntropyLoss,DiceCeLoss,DiceLoss,center_alignment_loss,KDLoss,mmd_loss
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from PIL import ImageOps, Image

dicece_loss = DiceCeLoss(4)

def tensor_rot_90(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).transpose(2, 3)
    else:
	    return x.flip(2).transpose(1, 2)
def tensor_rot_180(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).flip(2)
    else:
	    return x.flip(2).flip(1)
def tensor_flip_2(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(2)
    else:
	    return x.flip(1)
def tensor_flip_3(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3)
    else:
	    return x.flip(2)

def tensor_rot_270(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.transpose(2, 3).flip(3)
    else:
        return x.transpose(1, 2).flip(2)
def rotate_single_random(img):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        x = torch.transpose(img, 1, 2)
        img = torch.reshape(x, new_shape)
    label = np.random.randint(0, 4, 1)[0]
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img,label
def apply_gaussian_blur(tensor, kernel_size=5, sigma=1):
    # Ensure the tensor is in the right shape: [1, c, w, h]
    assert tensor.dim() == 4, "Input tensor must have 4 dimensions [1, c, w, h]"
    
    # Convert tensor to numpy array and transpose to [w, h, c]
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Apply Gaussian blur to each channel
    blurred_np = np.zeros_like(tensor_np)
    for i in range(tensor_np.shape[2]):
        blurred_np[:, :, i] = cv2.GaussianBlur(tensor_np[:, :, i], (kernel_size, kernel_size), sigma)
    
    # Convert back to tensor and permute to [1, c, w, h]
    blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1).unsqueeze(0)
    
    return blurred_tensor
def apply_color_distortion(tensor, brightness=0.5, contrast=0.5, saturation=0.5):
    # Ensure the tensor is in the right shape: [1, c, w, h]
    assert tensor.dim() == 4, "Input tensor must have 4 dimensions [1, c, w, h]"
    
    # Convert tensor to [c, w, h] for torchvision transforms
    tensor = tensor.squeeze(0)
    
    # Define the color distortion transform
    color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    
    # Apply the color distortion
    distorted_tensor = color_jitter(tensor)
    
    # Add the batch dimension back
    distorted_tensor = distorted_tensor.unsqueeze(0)
    
    return distorted_tensor
def apply_reflection(tensor, horizontal=True, vertical=True):
    # Ensure the tensor is in the right shape: [1, c, w, h]
    assert tensor.dim() == 4, "Input tensor must have 4 dimensions [1, c, w, h]"
    
    # Convert tensor to numpy array and transpose to [w, h, c]
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Apply reflection
    if horizontal:
        tensor_np = np.fliplr(tensor_np)
    if vertical:
        tensor_np = np.flipud(tensor_np)
    
    # Convert back to tensor and permute to [1, c, w, h]
    reflected_tensor = torch.from_numpy(tensor_np).permute(2, 0, 1).unsqueeze(0)
    
    return reflected_tensor

def rotate_single_with_label(img, label):
    # x_shape = list(img.shape)
    # if(len(x_shape) == 5):
    #     [N, C, D, H, W] = x_shape
    #     new_shape = [N*D, C, H, W]
    #     x = torch.transpose(img, 1, 2)
    #     img = torch.reshape(x, new_shape)
    # label = np.random.randint(0, 4, 1)[0]
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    elif label == 4:
        img = tensor_flip_2(img)
    elif label == 5:
        img = tensor_flip_3(img)
    elif label == 6:
        img = apply_gaussian_blur(img)
    elif label == 7:
        img = apply_color_distortion(img)
    elif label == 8:
        img = apply_reflection(img)
    else:
        img = img
    return img

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=0)
    return entropy, avg_logits

class TTA(nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
    def forward(self, x):
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def adapt_single(self, image):
        self.model.eval()
        for iteration in range(1):
            inputs = []
            for i in range(9):
                inputs.append(rotate_single_with_label(image, i))
            inputs = torch.stack(inputs).cuda()
            inputs = inputs.squeeze(dim=1)
            self.optimizer.zero_grad()
            # print(inputs.shape,'123')

            outputs = self.model(inputs)
            # loss, logits = marginal_entropy(outputs)
            # loss = loss.mean() / 320.0
            loss = self.entropy(outputs)
            loss = loss.mean()
            print(loss, '136')
            loss.backward()
            self.optimizer.step()

    def test_single(self, image):
        self.model.eval()
        with torch.no_grad():
            inputs = image
            outputs = self.model(inputs.cuda())
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        inputs = []
        for i in range(5):
            inputs.append(rotate_single_with_label(x, i))
        inputs.append(x)
        inputs = torch.stack(inputs)
        inputs = inputs.squeeze(dim=1)
        model.train()
        y = model(inputs)
        # print(y.shape,y[-1:,:].shape)
        return y[-1:,:]

        # optimizer.zero_grad()
        # # loss = self.entropy(y)
        # loss, logits = marginal_entropy(y)
        # loss = loss.mean() / 1.0
        # # print(loss, '136')
        # loss.backward()
        # self.optimizer.step()

        # # self.adapt_single(x)
        # output = model(x)
        # return output

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
            



 


@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    n, c, h, w =  x.shape
    en = (-0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum()-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum())
    en = en / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return en
def weight_symmetric_cross_entropy(x, x_ema, weights):# -> torch.Tensor:
    n, c, h, w =  x.shape
    assert len(weights) == n
    en = 0.
    for i in range(n):
        en += weights[i]*(-0.5*(x_ema[i:i+1].softmax(1) * x[i:i+1].log_softmax(1)).sum()-0.5*(x[i:i+1].softmax(1) * x_ema[i:i+1].log_softmax(1)).sum())
    en = en / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return en
@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1


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


import numpy as np
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) 

