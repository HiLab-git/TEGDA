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
from dataloaders.mms2d import MMS2D
from networks.net_factory_3d import net_factory_3d
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.calculate_metrics import evaluate_predictions
from val_3D import test_all_case
import csv
import math
import SimpleITK as sitk
from pymic.util.evaluation_seg import get_multi_class_evaluation_score
from sota._tta2d import *
from sota.aetta2d import AETTA
from sota.aetta4seg import AETTA4Seg
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/mms2d', help='Name of Experiment')
parser.add_argument('--source_domain', type=str,
                    default='A', help='The source domain')
parser.add_argument('--source_checkpoint', type=str,
                    default='/mnt/data1/ZhouFF/TTA4MIS/model/mms2d_Fully_Supervised_A/unet/source-A-model-latest.pth', help='The source domain checkpoint')
parser.add_argument('--target_domain', type=str,
                    default='B', help='The source domain')
parser.add_argument('--TTA_method', type=str,
                    default='source_test', help='The TTA methods')
parser.add_argument('--num_class', type=int,
                    default='4', help='The number of class')
parser.add_argument('--exp', type=str,
                    default='mms2d_A2B', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--iterations', type=int,
                    default=1, help='maximum epoch number to test')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--patch_size', type=list,  default=[320, 320],
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
    elif TTA_method == "meantimgupdate":
        logging.info("test-time adaptation: meantimgupdate")
        model = setup_meantimgupdate(base_model)
    elif TTA_method == "sitta":
        logging.info("test-time adaptation:sitta ")
        model = setup_sitta(base_model)
    elif TTA_method == "vptta":
        logging.info("test-time adaptation:vptta ")
        model = setup_vptta(base_model)
    elif TTA_method == "wjh":
        logging.info("test-time adaptation:wjh ")
        model = setup_wjh01(base_model)
    else:
        raise "no specific method of {}".format(TTA_method)
    return model


def online_evaluation_slice(name, label, pred):
    slice_results = []
    for batch_idx in range(pred.shape[0]):
        slice_dice_1 = get_multi_class_evaluation_score(s_volume=pred[batch_idx],g_volume=label[batch_idx],label_list=[1],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
        slice_dice_2 = get_multi_class_evaluation_score(s_volume=pred[batch_idx],g_volume=label[batch_idx],label_list=[2],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
        slice_dice_3 = get_multi_class_evaluation_score(s_volume=pred[batch_idx],g_volume=label[batch_idx],label_list=[3],fuse_label=True,spacing=[1.0,1.0,1.0],metric='dice')[0]
        Average_dice = (slice_dice_1+slice_dice_2+slice_dice_3)/3
        slice_dice_1 = round(slice_dice_1*100,2)
        slice_dice_2 = round(slice_dice_2*100,2)
        slice_dice_3 = round(slice_dice_3*100,2)
        Average_dice = round(Average_dice*100,2)
        
        # print(f'Ground Truth Dice of {name}: WT-{slice_dice_1},TC-{slice_dice_2},ET-{slice_dice_2},Avg-{Average_dice}')
        case_result = {'name':name[batch_idx],'class_1_dice':slice_dice_1,'class_2_dice':slice_dice_2,'class_3_dice':slice_dice_3,'Avg_dice':Average_dice}
        slice_results.append(case_result)
    
    return slice_results

    
def test_batch(net, image, num_classes):
    image = image.cuda().float()
    # with torch.no_grad():
    #     y1 = net(image)
    #     y = torch.argmax(y1, dim=1)
    y1 = net(image)
    y = torch.argmax(y1, dim=1)
    label = y.cpu().numpy()
    # print(label.shape)
    # score_map /= np.expand_dims(cnt, axis=0)  # 平均化
    # label_map = np.argmax(score_map, axis=0)  # 取最大值对应的类别

    return label

def train(args, snapshot_path):
    train_data_path = args.root_path + '/' + args.target_domain
    batch_size = args.batch_size
    num_classes = args.num_class
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_class)
    db_train = MMS2D(base_dir=train_data_path,
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
    
    eval_model = deepcopy(model)
    
    model = setup_TTA_model(model, args.TTA_method)


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    save_output_dir = os.path.join(snapshot_path, 'prediction'+args.TTA_method)
    os.makedirs(save_output_dir,exist_ok=True)
    iter_num = 0
    results = []
    eval = AETTA()
    for i_batch, sampled_batch in enumerate(trainloader):
        name, volume_batch, label_batch = sampled_batch['name'], sampled_batch['image'], sampled_batch['label']
        # volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        output = test_batch(model, volume_batch, num_classes=args.num_class)
        
        # 这里需要设置batch size为1，然后针对每个slice进行测试
        case_result = online_evaluation_slice(name,label_batch[:,0,0],output)[0]
        # est_1, est_2, est_3, est_avg, mismatch_mask, entropy, var, acc,est_avg_noweight,est_avg_weight_inv = eval.aetta(input=volume_batch,pred=output,model=eval_model, multi_eval=True)
        est_1, est_2, est_3, est_avg, mismatch_mask, entropy, var, acc = eval.aetta(input=volume_batch,pred=output,model=eval_model, multi_eval=True)
        case_result['acc'] = acc
        case_result['var'] = var
        case_result['entropy'] = entropy
        # case_result['adi'] = est_avg_noweight
        case_result['adiu'] = est_avg
        # case_result['adiu_inv'] = est_avg_weight_inv
        results.append(case_result)
        
        # print(output.shape)
        # print("Adaptated Case:",name)
        # print(model.state_dict())
        # slice_results = online_evaluation_slice(name,label_batch[0][0],output)
        # results.extend(slice_results)
        ###################### save output ##################
        for i in range(output.shape[0]):
            test_save_path = os.path.join(save_output_dir, sampled_batch['name'][i])
            prd_itk = sitk.GetImageFromArray(output[i].astype(np.float32))
            sitk.WriteImage(prd_itk, test_save_path)
            
        iter_num = iter_num + 1


    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(iter_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    
    result_df = pd.DataFrame(results)
    WT_mean=round(result_df['class_1_dice'].mean(),2)
    WT_std=round(result_df['class_1_dice'].std(),2)
    TC_mean=round(result_df['class_2_dice'].mean(),2)
    TC_std=round(result_df['class_2_dice'].std(),2)
    ET_mean=round(result_df['class_3_dice'].mean(),2)
    ET_std=round(result_df['class_3_dice'].std(),2)
    Avg_mean=round(result_df['Avg_dice'].mean(),2)
    Avg_std=round(result_df['Avg_dice'].std(),2)
    mean_std_row = pd.DataFrame({'name': ['mean'], 
                             'class_1_dice':[f'{WT_mean}±{WT_std}'],
                             'class_2_dice':[f'{TC_mean}±{TC_std}'],
                             'class_3_dice':[f'{ET_mean}±{ET_std}'],
                             'Avg_dice':[f'{Avg_mean}±{Avg_std}']})

    # # 添加平均值和标准差到结果 DataFrame
    result_df = pd.concat([mean_std_row,result_df], ignore_index=True)

    # # 将结果写入输出 CSV 文件
    output_dir = snapshot_path + "/final_result_slices.csv"
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
