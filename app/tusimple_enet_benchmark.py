from essentials.add_module import set_working_directory
set_working_directory()

import os
import time
import ujson as json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import MeanShift
from neural_networks.enet import Enet
from utils.path import Path
from data.datasets.lane_dataset import TuSimpleDataset


def fit_lanes(inst_pred):
    assert inst_pred.dim() == 2

    inst_pred_expand = inst_pred.view(-1)

    inst_unique = torch.unique(inst_pred_expand)

    lanes = []
    for inst_idx in inst_unique:
        if inst_idx != 0:
            lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())

    curves = []
    for lane in lanes:
        pts = lane
        curve = np.polyfit(pts[:, 0], pts[:, 1], 3)
        curves.append(curve)

    return curves


def sample_from_curve(curves, inst_pred, y_sample):
    h, w = inst_pred.shape
    curves_pts = []
    for param in curves:
        fy = np.poly1d(param)
        x_sample = fy(y_sample)

        index = np.where(np.logical_or(x_sample < 0, x_sample >= w))
        x_sample[index] = -2

        index = np.where((inst_pred[y_sample, x_sample] == 0).cpu().numpy())
        x_sample[index] = -2

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


def main():
    # Test config
    batch_size = 64
    num_workers = 4
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    model_name = 'ckpt_2025-05-22_22-38-37_epoch-10.pth'
    tag = 'enet_benchmark'

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Using", torch.cuda.device_count(), "GPUs")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print("Batch size:", batch_size)

    # Get paths
    path = Path()
    test_data_dir = path.test_data
    model_path = path.get_model(model_name)

    # Load dataset
    test_set = TuSimpleDataset(test_data_dir, phase='test')
    num_test = len(test_set)
    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Loaded data from', test_data_dir)

    # Get image dimensions
    _, h, w = test_set[0]['input_tensor'].shape
    
    # TuSimple Benchmark settings
    y_start = np.round(160 * h / 720.)
    y_stop = np.round(710 * h / 720.)
    y_num = 56
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)

    # Initialize network
    with torch.no_grad():
        net = Enet()
        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # Testing loop
        data_iter = iter(testset_loader)
        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        output_list = list()

        for step in range(num_test):
            time_run = time.time()
            time_fp = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                break

            input_batch = batch['input_tensor'].to(device)
            raw_file_batch = batch['raw_file']
            path_batch = batch['path']

            # Forward pass
            embeddings, logit = net(input_batch)
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])
            time_fp = time.time() - time_fp

            # Clustering
            time_clst = time.time()
            pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            time_clst = time.time() - time_clst

            # Curve fitting
            time_fit = time.time()
            for idx in range(batch_size):
                pred_inst = pred_insts[idx]
                raw_file = raw_file_batch[idx]

                # Direct curve fitting
                curves_param = fit_lanes(pred_inst)
                curves_pts_pred = sample_from_curve(curves_param, pred_inst, y_sample)

                time_fit = time.time() - time_fit
                time_run = time.time() - time_run

                # Generate JSON entry
                json_entry = generate_json_entry(curves_pts_pred, y_sample, raw_file, (h, w), time_run)
                output_list.append(json_entry)

                # Update timing statistics
                time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                time_ct += 1

                if step % 50 == 0:
                    time_ct = 0

                print('{}  {}  Step:{}  Time:{:5.1f}  '
                      'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
                      .format(train_start_time, tag, step, time_run*1000,
                             time_run_avg*1000, time_fp_avg * 1000, time_clst_avg * 1000, time_fit_avg * 1000,
                             int(1/(time_run_avg + 1e-9))))

        # Save results
        output_file = path.get_output(f'test_pred-{train_start_time}-{model_name}-{tag}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for item in output_list:
                json.dump(item, f)
                f.write('\n')
        print(f'Results saved to {output_file}')


if __name__ == '__main__':
    main()
