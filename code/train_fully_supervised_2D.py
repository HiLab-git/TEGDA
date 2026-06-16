import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
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
from val_3D import test_all_case
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/mms2d', help='Name of Experiment')
parser.add_argument('--source_domain', type=str,
                    default='A', help='The source domain')
parser.add_argument('--num_class', type=int,
                    default='4', help='The number of class')
parser.add_argument('--exp', type=str,
                    default='mms2d_Fully_Supervised_A', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[320, 320],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=100,
                    help='labeled data')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path + '/' + args.source_domain
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_class = args.num_class
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_class)
    db_train = MMS2D(base_dir=train_data_path,
                         split='train',
                         num=args.labeled_num,
                         )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    # source_checkpoint = '/mnt/data1/ZhouFF/TTA4MIS/model/BraTs2023_Fully_Supervised_GLI_norm/unet_3D/iter_27000_dice_0.7735.pth'
    # checkpoint = torch.load(source_checkpoint, map_location='cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print(f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")
    # logging.info(f"Successfully loaded {len(pretrained_dict)} layers from checkpoint, ignored {len(checkpoint) - len(pretrained_dict)} mismatched layers.")


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_class)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda()
            
            # print(sampled_batch['name'],volume_batch.dtype, label_batch.dtype)
            # print(volume_batch.min(),volume_batch.max())

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch.squeeze().long())
            loss_dice = dice_loss(outputs_soft, label_batch.squeeze(1).long())
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            
            # if iter_num % 200 == 0:
            #     predicted_label = outputs_soft.argmax(dim=1, keepdim=True)
            #     foreground_slices = torch.where(label_batch > 0)[4].unique()
            #     if len(foreground_slices) > 0:
            #         selected_slice = random.choice(foreground_slices).item()
            #     else:
            #         selected_slice = random.randint(0, volume_batch.shape[4] - 1)
            #     print(f"Selected slice: {selected_slice}")

            #     image = volume_batch[0, 0:1, :, :, selected_slice:selected_slice+1].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     predicted_image = predicted_label[0, :, :, :, selected_slice:selected_slice+1].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(predicted_image*80, 5, normalize=False)
            #     writer.add_image('train/Predicted_label', grid_image, iter_num)

            #     groundtruth_image = label_batch[0, :, :, :, selected_slice:selected_slice+1].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(groundtruth_image*80, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)
                # print(predicted_image.max(),predicted_image.min(),groundtruth_image.max(),groundtruth_image.min())

            # if iter_num > 0 and iter_num % 1000 == 0:
            #     model.eval()
            #     avg_metric = test_all_case(
            #         model, train_data_path, test_list="valid.csv", num_classes=args.num_class, patch_size=args.patch_size,
            #         stride_xy=320, stride_z=320)
            #     if avg_metric[:, 0].mean() > best_performance:
            #         best_performance = avg_metric[:, 0].mean()
            #         save_mode_path = os.path.join(snapshot_path,
            #                                       'iter_{}_dice_{}.pth'.format(
            #                                           iter_num, round(best_performance, 4)))
            #         save_best = os.path.join(snapshot_path,
            #                                  '{}_best_model.pth'.format(args.model))
            #         torch.save(model.state_dict(), save_mode_path)
            #         torch.save(model.state_dict(), save_best)

            #     writer.add_scalar('info/val_dice_score',
            #                       avg_metric[0, 0], iter_num)
            #     writer.add_scalar('info/val_hd95',
            #                       avg_metric[0, 1], iter_num)
            #     logging.info(
            #         'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
            #     model.train()

            if iter_num % 10000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


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
