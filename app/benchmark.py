from essentials.add_module import set_working_directory
set_working_directory()

import os
import time
import argparse
import ujson as json
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from neural_networks.enet import Enet
from data.datasets.lane_dataset import TuSimpleDataset
from utils.path import Path

import numpy as np
import cv2 as cv2
import torch
from sklearn.cluster import MeanShift


def fit_lanes(inst_pred):
    """

    :param inst_pred: lane instances prediction map, support single image
    :return: A list of each curve's parameter
    """
    assert inst_pred.dim() == 2

    h, w = inst_pred.shape


    inst_pred_expand = inst_pred.view(-1)

    inst_unique = torch.unique(inst_pred_expand)

    # extract points coordinates for each lane
    lanes = []
    for inst_idx in inst_unique:
        if inst_idx != 0:

            # lanes.append(torch.nonzero(torch.tensor(inst_pred == inst_idx).byte()).numpy())

            lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())

    curves = []
    for lane in lanes:
        pts = lane

        # fitting each lane
        curve = np.polyfit(pts[:, 0], pts[:, 1], 3)
        curves.append(curve)

    return curves


def sample_from_curve(curves, inst_pred, y_sample):
    """

    :param curves: A list of each curve's parameter
    :param inst_pred: lane instances prediction map, support single image
    :return: A list of sampled points on each curve
    """
    h, w = inst_pred.shape
    curves_pts = []
    for param in curves:
        # use new curve function f(y) to calculate x values
        fy = np.poly1d(param)
        x_sample = fy(y_sample)

        '''Filter out points beyond image boundaries'''
        index = np.where(np.logical_or(x_sample < 0, x_sample >= w))
        x_sample[index] = -2

        '''Filter out points beyond predictions'''
        # may filter out bad point, but can also drop good point at the edge
        index = np.where((inst_pred[y_sample, x_sample] == 0).cpu().numpy())
        x_sample[index] = -2

        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)

    return curves_pts


def sample_from_IPMcurve(curves, pred_inst_IPM, y_sample):
    """

    :param curves: A list of each curve's parameter
    :param inst_pred: lane instances prediction map, support single image
    :return: A list of sampled points on each curve
    """
    h, w = pred_inst_IPM.shape
    curves_pts = []
    for param in curves:
        # use new curve function f(y) to calculate x values
        fy = np.poly1d(param)
        x_sample = fy(y_sample)

        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)

    return curves_pts


def generate_json_entry(curves_pts_pred, y_sample, raw_file, size, run_time):
    h, w = size

    lanes = []
    for curve in curves_pts_pred:
        index = np.where(curve[:, 0] > 0)
        curve[index, 0] = curve[index, 0] * 720. / h

        x_list = np.round(curve[:, 0]).astype(np.int32).tolist()
        lanes.append(x_list)

    entry_dict = dict()

    entry_dict['lanes'] = lanes
    entry_dict['h_sample'] = np.round(y_sample * 720. / h).astype(np.int32).tolist()
    entry_dict['run_time'] = int(np.round(run_time * 1000))
    entry_dict['raw_file'] = raw_file

    return entry_dict

def cluster_embed(embeddings, preds_bin, band_width):
    c = embeddings.shape[1]
    n, _, h, w = preds_bin.shape
    preds_bin = preds_bin.view(n, h, w)
    preds_inst = torch.zeros_like(preds_bin)
    for idx, (embedding, bin_pred) in enumerate(zip(embeddings, preds_bin)):
        # Convert bin_pred to boolean tensor
        bin_pred_bool = bin_pred.bool()
        embedding_fg = torch.transpose(torch.masked_select(embedding, bin_pred_bool).view(c, -1), 0, 1)

        clustering = MeanShift(bandwidth=band_width, bin_seeding=True, min_bin_freq=100).fit(embedding_fg.cpu().detach().numpy())

        preds_inst[idx][bin_pred_bool] = torch.from_numpy(clustering.labels_).cuda() + 1

    return preds_inst

def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


if __name__ == '__main__':
    '''Test config'''
    batch_size = 1
    num_workers = 4
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")
        print("Let's use CPU")
    print("Batch size: %d" % batch_size)

    output_dir = Path().get_output("tusimple_enet_benchmark")
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    data_dir = Path().test_data

    test_set = TuSimpleDataset(data_dir, phase='test', size=(512,288))

    num_test = len(test_set)

    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'test': testset_loader}
    phase = 'test'

    print('Finish loading data from %s' % data_dir)

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    _, h, w = test_set[0]['input_tensor'].shape

    # for IPM (Inverse Projective Mapping)
    src = np.float32([[0.35 * (w - 1), 0.34 * (h - 1)], [0.65 * (w - 1), 0.34 * (h - 1)],
                      [0. * (w - 1), h - 1], [1. * (w - 1), h - 1]])
    dst = np.float32([[0. * (w - 1), 0. * (h - 1)], [1.0 * (w - 1), 0. * (h - 1)],
                      [0.4 * (w - 1), (h - 1)], [0.60 * (w - 1), (h - 1)]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # y_start, y_stop and y_num is calculated according to TuSimple Benchmark's setting
    y_start = np.round(160 * h / 720.)
    y_stop = np.round(710 * h / 720.)
    y_num = 56
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)
    x_sample = np.zeros_like(y_sample, dtype=np.float32) + w // 2
    c_sample = np.ones_like(y_sample, dtype=np.float32)
    xyc_sample = np.vstack((x_sample, y_sample, c_sample))

    xyc_IPM = M.dot(xyc_sample).T

    y_IPM = []
    for pt in xyc_IPM:
        y = np.round(pt[1] / pt[2])
        y_IPM.append(y)
    y_IPM = np.array(y_IPM)

    '''Forward propogation'''
    with torch.no_grad():
        net = Enet()

        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        checkpoint = torch.load(Path().get_model("ckpt_2025-05-22_22-38-37_epoch-10.pth"))
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        step = 0
        epoch = 1
        print()

        data_iter = {'test': iter(dataloaders['test'])}
        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        output_list = list()
        for step in range(num_test):
            time_run = time.time()
            time_fp = time.time()

            '''load dataset'''
            try:
                batch = next(data_iter[phase])
            except StopIteration:
                break

            input_batch = batch['input_tensor']
            raw_file_batch = batch['raw_file']
            path_batch = batch['path']

            input_batch = input_batch.to(device)

            # forward
            embeddings, logit = net(input_batch)

            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])

            time_fp = time.time() - time_fp

            '''sklearn mean_shift'''
            time_clst = time.time()
            pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            time_clst = time.time() - time_clst

            '''Curve Fitting'''
            time_fit = time.time()
            ipm = False
            for idx in range(batch_size):
                input_rgb = input_batch[idx]  # for each image in a batch
                raw_file = raw_file_batch[idx]
                pred_inst = pred_insts[idx]
                path = path_batch[idx]
                if ipm:
                    '''Fit Curve after IPM(Inverse Perspective Mapping)'''
                    pred_inst_IPM = cv2.warpPerspective(pred_inst.cpu().numpy().astype('uint8'), M, (w, h),
                                                        flags=cv2.INTER_NEAREST)
                    pred_inst_IPM = torch.from_numpy(pred_inst_IPM)

                    curves_param = fit_lanes(pred_inst_IPM)
                    curves_pts_IPM = sample_from_IPMcurve(curves_param, pred_inst_IPM, y_IPM)
                    
                    curves_pts_pred = []
                    for xy_IPM in curves_pts_IPM:  # for each lane in a image
                        n, _ = xy_IPM.shape
                    
                        c_IPM = np.ones((n, 1))
                        xyc_IPM = np.hstack((xy_IPM, c_IPM))
                        xyc_pred = M_inv.dot(xyc_IPM.T).T
                    
                        xy_pred = []
                        for pt in xyc_pred:
                            x = np.round(pt[0] / pt[2]).astype(np.int32)
                            y = np.round(pt[1] / pt[2]).astype(np.int32)
                            if 0 <= y < h and 0 <= x < w:  # and pred_inst[y, x]
                                xy_pred.append([x,y])
                            else:
                                xy_pred.append([-2, y])
                    
                        xy_pred = np.array(xy_pred, dtype=np.int32)
                        curves_pts_pred.append(xy_pred)
                else:
                    '''Directly fit curves on original images'''
                    curves_param = fit_lanes(pred_inst)
                    curves_pts_pred=sample_from_curve(curves_param,pred_inst, y_sample)

                time_fit = time.time() - time_fit
                time_run = time.time() - time_run

                # Generate Json file to be evaluated by TuSimple Benchmark official eval script
                json_entry = generate_json_entry(curves_pts_pred, y_sample, raw_file, (h, w), time_run)
                output_list.append(json_entry)

                time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                time_ct += 1

                if step % 50 == 0:  # Change the coefficient to filter the value
                    time_ct = 0


                print('{}  Step:{}  Time:{:5.1f}  '
                      'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
                      .format(train_start_time, step, time_run*1000,
                             time_run_avg*1000, time_fp_avg * 1000, time_clst_avg * 1000, time_fit_avg * 1000,
                             int(1/(time_run_avg + 1e-9))))

            '''Write to Tensorboard Summary'''
            # num_images = 3
            # inputs_images = (input_batch + VGG_MEAN)[:num_images, [2, 1, 0], :, :]  # .byte()
            # writer.add_images('image', inputs_images, step)
            # #
            # writer.add_images('Bin Pred', pred_bin_batch[:num_images], step)
            # #
            # labels_bin_img = labels_bin_batch.view(labels_bin_batch.shape[0], 1, labels_bin_batch.shape[1], labels_bin_batch.shape[2])
            # writer.add_images('Bin Label', labels_bin_img[:num_images], step)
            #
            # embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5
            # # print(torch.min(embedding_img).item(), torch.max(embedding_img).item())
            # writer.add_images('Embedding', embedding_img, step)

        with open(f'{output_dir}/test_pred-{train_start_time}.json', 'w') as f:
            for item in output_list:
                json.dump(item, f)  # , indent=4, sort_keys=True
                f.write('\n')
