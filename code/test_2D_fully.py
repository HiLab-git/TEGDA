import argparse
import os
import shutil

import h5py
import csv
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() == 0 and gt.sum() == 0:
        asd = 0.0
        hd95 = 0.0
    elif pred.sum() > 0 and gt.sum() > 0:
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    else:
        asd = 5.0
        hd95 = 5.0
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    score_all_data_0 = []
    name_score_list_0= []
    score_all_data_1 = []
    name_score_list_1= [] 
    score_all_data_2 = []
    name_score_list_2= [] 
    for case in tqdm(image_list):
        name = case.split('.')[0]
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        score_vector_0 = [first_metric[0], second_metric[0], third_metric[0]]
        score_vector_1 = [first_metric[1], second_metric[1], third_metric[1]]
        score_vector_2 = [first_metric[2], second_metric[2], third_metric[2]]
        if(FLAGS.num_classes > 2):
            score_vector_0.append(np.asarray(score_vector_0).mean())
            score_vector_1.append(np.asarray(score_vector_1).mean())
            score_vector_2.append(np.asarray(score_vector_2).mean())
        score_all_data_0.append(score_vector_0)
        score_all_data_1.append(score_vector_1)
        score_all_data_2.append(score_vector_2)
        name_score_list_0.append([name] + score_vector_0)
        name_score_list_1.append([name] + score_vector_1)
        name_score_list_2.append([name] + score_vector_2)
    score_all_data_0 = np.asarray(score_all_data_0)
    score_mean0 = score_all_data_0.mean(axis = 0)
    score_std0  = score_all_data_0.std(axis = 0)
    name_score_list_0.append(['mean'] + list(score_mean0))
    name_score_list_0.append(['std'] + list(score_std0))
    score_all_data_1 = np.asarray(score_all_data_1)
    score_mean1 = score_all_data_1.mean(axis = 0)
    score_std1  = score_all_data_1.std(axis = 0)
    name_score_list_1.append(['mean'] + list(score_mean1))
    name_score_list_1.append(['std'] + list(score_std1))
    score_all_data_2 = np.asarray(score_all_data_2)
    score_mean2 = score_all_data_2.mean(axis = 0)
    score_std2  = score_all_data_2.std(axis = 0)
    name_score_list_2.append(['mean'] + list(score_mean2))
    name_score_list_2.append(['std'] + list(score_std2))
    # save the result as csv 
    score_csv0 = "{0:}/aa_test_{1:}_all.csv".format(test_save_path, 'dice')
    score_csv1 = "{0:}/aa_test_{1:}_all.csv".format(test_save_path, 'assd')
    score_csv2 = "{0:}/aa_test_{1:}_all.csv".format(test_save_path, 'hd95')
    with open(score_csv0, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                        quotechar='"',quoting=csv.QUOTE_MINIMAL)
        head = ['image'] + ["class_{0:}".format(i) for i in range(1,FLAGS.num_classes)]
        if(FLAGS.num_classes > 2):
            head = head + ["average"]
        csv_writer.writerow(head)
        for item in name_score_list_0:
            csv_writer.writerow(item)
    with open(score_csv1, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                        quotechar='"',quoting=csv.QUOTE_MINIMAL)
        head = ['image'] + ["class_{0:}".format(i) for i in range(1,FLAGS.num_classes)]
        if(FLAGS.num_classes > 2):
            head = head + ["average"]
        csv_writer.writerow(head)
        for item in name_score_list_1:
            csv_writer.writerow(item)
    with open(score_csv2, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                        quotechar='"',quoting=csv.QUOTE_MINIMAL)
        head = ['image'] + ["class_{0:}".format(i) for i in range(1,FLAGS.num_classes)]
        if(FLAGS.num_classes > 2):
            head = head + ["average"]
        csv_writer.writerow(head)
        for item in name_score_list_2:
            csv_writer.writerow(item)
    print("Test data: {0} mean {1}".format('dice', score_mean0))
    print("Test data: {0} std {1}".format('dice', score_std0))
    print("Test data: {0} mean {1}".format('assd', score_mean1))
    print("Test data: {0} std {1}".format('assd', score_std1))
    print("Test data: {0} mean {1}".format('hd95', score_mean2))
    print("Test data: {0} std {1}".format('hd95', score_std2))


    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    # return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
    # print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
