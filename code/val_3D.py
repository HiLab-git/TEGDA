import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import csv
import torchio as tio


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    c, w, h, d = image.shape  # 4D: c, w, h, d

    # 如果 image 的尺寸小于 patch_size，则进行填充
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
                    # 使用 softmax 获取概率
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0]  # 去除 batch 维度 (num_classes, w, h, d)

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map /= np.expand_dims(cnt, axis=0)  # 平均化
    label_map = np.argmax(score_map, axis=0)  # 取最大值对应的类别

    if add_pad:
        # 如果有填充，则去掉填充部分
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map
    
# def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
#     c, w, h, d = image.shape

#     # if the size of image is less than patch_size, then padding it
#     add_pad = False
#     if w < patch_size[0]:
#         w_pad = patch_size[0]-w
#         add_pad = True
#     else:
#         w_pad = 0
#     if h < patch_size[1]:
#         h_pad = patch_size[1]-h
#         add_pad = True
#     else:
#         h_pad = 0
#     if d < patch_size[2]:
#         d_pad = patch_size[2]-d
#         add_pad = True
#     else:
#         d_pad = 0
#     wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
#     hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
#     dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
#     if add_pad:
#         image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
#                                (dl_pad, dr_pad)], mode='constant', constant_values=0)
#     ww, hh, dd = image.shape

#     sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
#     sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
#     sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
#     # print("{}, {}, {}".format(sx, sy, sz))
#     score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
#     cnt = np.zeros(image.shape).astype(np.float32)

#     for x in range(0, sx):
#         xs = min(stride_xy*x, ww-patch_size[0])
#         for y in range(0, sy):
#             ys = min(stride_xy * y, hh-patch_size[1])
#             for z in range(0, sz):
#                 zs = min(stride_z * z, dd-patch_size[2])
#                 test_patch = image[xs:xs+patch_size[0],
#                                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
#                 test_patch = np.expand_dims(np.expand_dims(
#                     test_patch, axis=0), axis=0).astype(np.float32)
#                 test_patch = torch.from_numpy(test_patch).cuda()

#                 with torch.no_grad():
#                     y1 = net(test_patch)
#                     # ensemble
#                     y = torch.softmax(y1, dim=1)
#                 y = y.cpu().data.numpy()
#                 y = y[0, :, :, :, :]
#                 score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
#                     = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
#                 cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
#                     = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
#     score_map = score_map/np.expand_dims(cnt, axis=0)
#     label_map = np.argmax(score_map, axis=0)

#     if add_pad:
#         label_map = label_map[wl_pad:wl_pad+w,
#                               hl_pad:hl_pad+h, dl_pad:dl_pad+d]
#         score_map = score_map[:, wl_pad:wl_pad +
#                               w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
#     return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = []
        reader = csv.reader(f)
        next(reader)  # 跳过header
        for row in reader:
            image_list.append({
                'image': row[0],
                'image2': row[1],
                'image3': row[2],
                'image4': row[3],
                'label': row[4]
            })
        total_metric = np.zeros((num_classes-1, 2))
        print("Validation begin")
        transform = tio.Compose([
                tio.RescaleIntensity(out_min_max=(0, 1))
            ])
        for image_path in tqdm(image_list):
            images = []
            for key in ['image', 'image2', 'image3', 'image4']:
                image = sitk.ReadImage(image_path[key])
                image_array = sitk.GetArrayFromImage(image)
                images.append(image_array)
            image = np.stack(images, axis=0)  # [channels, height, width, depth]
            label = sitk.ReadImage(image_path['label'])
            label_array = sitk.GetArrayFromImage(label)
            label_array = label_array.reshape(1, label_array.shape[0], label_array.shape[1], label_array.shape[2])

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image), 
                label=tio.LabelMap(tensor=label_array))
            transformed_subject = transform(subject)
            image = transformed_subject['image'].numpy()
            label_array = transformed_subject['label'].numpy()
            label_array = label_array[0]
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            for i in range(1, num_classes):
                total_metric[i-1, :] += cal_metric(label_array == i, prediction == i)
    print("Validation end")
    return total_metric / len(image_list)
