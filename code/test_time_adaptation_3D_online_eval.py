import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import copy
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2023 import BraTS2023
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from utils.calculate_metrics import evaluate_predictions
from val_3D import test_all_case
import csv
import math
import SimpleITK as sitk
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from sota._tta import *
from sota.adic import ADIC
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2023', help='Name of Experiment')
parser.add_argument('--source_domain', type=str,
                    default='BraTS_GLI', help='The source domain')
parser.add_argument('--source_checkpoint', type=str,
                    default='', help='The source domain checkpoint')
parser.add_argument('--target_domain', type=str,
                    default='BraTS_MEN', help='The source domain')
parser.add_argument('--TTA_method', type=str,
                    default='source_test', help='The TTA methods')
parser.add_argument('--num_class', type=int,
                    default='4', help='The number of class')
parser.add_argument('--exp', type=str,
                    default='BraTs2023_GLI2MEN', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--iterations', type=int,
                    default=1, help='maximum epoch number to test')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=100,
                    help='labeled data')

args = parser.parse_args()

def setup_TTA_model(base_model, TTA_method):
    if TTA_method == "source_test":
        logging.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif TTA_method == "norm":
        logging.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    elif TTA_method == "tent":
        logging.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    elif TTA_method == "cotta":
        logging.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    elif TTA_method == "sar":
        logging.info("test-time adaptation: SAR")
        model = setup_sar(base_model)
    elif TTA_method == "meant":
        logging.info("test-time adaptation: meant")
        model = setup_meant(base_model)
    elif TTA_method == "tegda":
        logging.info("test-time adaptation: tegda")
        model = setup_tegda(base_model)
    elif TTA_method == "sitta":
        logging.info("test-time adaptation: sitta")
        model = setup_sitta(base_model)
    elif TTA_method == "vptta":
        logging.info("test-time adaptation: VPTTA") 
        model = setup_vptta(base_model)
    else:
        raise "no specific method of {}".format(TTA_method)
    return model

def test_single_case_sliding_window(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    c, w, h, d = image.shape  # 4D: c, w, h, d

    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0

    # 计算左右填充
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    if add_pad:
        # 填充时需要考虑通道维度，因此对 (c, w, h, d) 进行填充
        image = np.pad(image, [(0, 0), (wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)

    _, ww, hh, dd = image.shape  # 填充后的尺寸

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ww, hh, dd), dtype=np.float32)  # 输出为(num_classes, w, h, d)
    cnt = np.zeros((ww, hh, dd), dtype=np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                
                # 提取 patch 并保留通道维度
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)  # 扩展 batch 维度
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0]  

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map /= np.expand_dims(cnt, axis=0)  # 平均化
    label_map = np.argmax(score_map, axis=0)  # 取最大值对应的类别

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map


def online_evaluation(name, label, pred):
    WT_dice = get_multi_class_evaluation_score(s_volume=pred,g_volume=label,label_list=[1,2,3],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
    TC_dice = get_multi_class_evaluation_score(s_volume=pred,g_volume=label,label_list=[2,3],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
    ET_dice = get_multi_class_evaluation_score(s_volume=pred,g_volume=label,label_list=[3],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
    Average_dice = (WT_dice + TC_dice + ET_dice)/3
    WT_dice = round(WT_dice*100,2)
    TC_dice = round(TC_dice*100,2)
    ET_dice = round(ET_dice*100,2)
    Average_dice = round(Average_dice*100,2)

    print(f'Ground Truth Dice: WT-{WT_dice},TC-{TC_dice},ET-{ET_dice},Avg-{Average_dice}')
    case_result = {'name':name,'WT_dice':WT_dice,'TC_dice':TC_dice,'ET_dice':ET_dice,'Avg_dice':Average_dice}
    
    return case_result
    
    
def test_single_case(net, image, num_classes):
    image = image.cuda().float()
    # with torch.no_grad():
    #     y1 = net(image)
    #     y = torch.argmax(y1, dim=1)
    y1 = net(image)
    y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()[0] 
    # print(label.shape)
    # score_map /= np.expand_dims(cnt, axis=0)  # 平均化
    # label_map = np.argmax(score_map, axis=0)  # 取最大值对应的类别

    return label

def train(args, snapshot_path):
    train_data_path = args.root_path + '/' + args.target_domain
    batch_size = args.batch_size
    num_classes = args.num_class
    model = net_factory_3d(net_type=args.model, in_chns=4, class_num=args.num_class)
    db_train = BraTS2023(base_dir=train_data_path,
                         split='all',
                         num=args.labeled_num
                         )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    checkpoint = torch.load(args.source_checkpoint, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")
    logging.info(f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")
    model = setup_TTA_model(model, args.TTA_method)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    save_output_dir = os.path.join(snapshot_path, 'prediction'+args.TTA_method)
    os.makedirs(save_output_dir,exist_ok=True)
    iter_num = 0
    results = []
    for i_batch, sampled_batch in enumerate(trainloader):
        name, volume_batch, label_batch = sampled_batch['name'][0], sampled_batch['image'], sampled_batch['label']
        # volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        output = test_single_case(model, volume_batch, num_classes=args.num_class)
        print("Adaptated Case:",name)
        # print(model.state_dict())
        case_result = online_evaluation(name,label_batch[0][0],output)
        results.append(case_result)
        ###################### save output ##################
        test_save_path = os.path.join(save_output_dir, sampled_batch['name'][0])
        prd_itk = sitk.GetImageFromArray(output.astype(np.float32))
        sitk.WriteImage(prd_itk, test_save_path)
        ###################### calculate ####################
        # loss_ce = ce_loss(output, label_batch.squeeze(1).long())
        # loss_dice = dice_loss(output, label_batch)
        # loss = 0.5 * (loss_dice + loss_ce)

        iter_num = iter_num + 1
        # writer.add_scalar('info/lr', lr_, iter_num)
        # writer.add_scalar('info/total_loss', loss, iter_num)
        # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
        # writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        # logging.info(
        #     'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
        #     (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
        # writer.add_scalar('loss/loss', loss, iter_num)

    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    
    result_df = pd.DataFrame(results)
    WT_mean=round(result_df['WT_dice'].mean(),2)
    WT_std=round(result_df['WT_dice'].std(),2)
    TC_mean=round(result_df['TC_dice'].mean(),2)
    TC_std=round(result_df['TC_dice'].std(),2)
    ET_mean=round(result_df['ET_dice'].mean(),2)
    ET_std=round(result_df['ET_dice'].std(),2)
    Avg_mean=round(result_df['Avg_dice'].mean(),2)
    Avg_std=round(result_df['Avg_dice'].std(),2)
    mean_std_row = pd.DataFrame({'name': ['mean'], 
                             'WT_dice':[f'{WT_mean}±{WT_std}'],
                             'TC_dice':[f'{TC_mean}±{TC_std}'],
                             'ET_dice':[f'{ET_mean}±{ET_std}'],
                             'Avg_dice':[f'{Avg_mean}±{Avg_std}']})

    # 添加平均值和标准差到结果 DataFrame
    result_df = pd.concat([mean_std_row,result_df], ignore_index=True)

    # 将结果写入输出 CSV 文件
    output_dir = snapshot_path + "/final_result.csv"
    result_df.to_csv(output_dir, index=False)
    # evaluate_predictions(save_output_dir,train_data_path+'/'+'preprocessed_data','seg','seg',args.num_class)


    logging.info("save model to {}".format(save_mode_path))
    writer.close()
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
