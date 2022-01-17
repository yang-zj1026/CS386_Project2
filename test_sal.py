import ipdb
import torch
import torch.nn.functional as F
import time
import os
import sys
import numpy as np
from numpy import nonzero
from imageio import imwrite

from utils import AverageMeter
from models.sal_losses import cc_score, nss_score, similarity, auc_judd, auc_shuff_np


def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) *
                        (255.0 / (data_max - data_min)),
                        0, 255).astype(np.uint8)
    return data_norm


def save_video_results(output_buffer, save_path):
    video_outputs = torch.stack(output_buffer)
    for i in range(video_outputs.size()[0]):
        save_name = os.path.join(save_path, 'pred_sal_{0:06d}.jpg'.format(i + 9))
        imwrite(save_name, normalize_data(video_outputs[i][0].numpy()))


def test(data_loader, model, opt):
    print('test')

    model.eval()

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end_time = time.time()
        output_buffer = []
        previous_video_id = ''
        cc = AverageMeter()
        nss = AverageMeter()
        sim = AverageMeter()
        auc_j = AverageMeter()

        for i, (data, targets, valid) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets['salmap'] = targets['salmap'].cuda()
                targets['binmap'] = targets['binmap'].cuda()
                valid['sal'] = valid['sal'].cuda()

            inputs = data['rgb']
            curr_batch_size = inputs.size()[0]
            targets['salmap'] = targets['salmap'].float()
            targets['binmap'] = targets['binmap'].float()
            valid['sal'] = valid['sal'].float()

            while inputs.size()[0] < opt.batch_size:
                inputs = torch.cat((inputs, inputs[0:1, :]), 0)
            while data['audio'].size(0) < opt.batch_size:
                data['audio'] = torch.cat((data['audio'], data['audio'][0:1, :]), 0)

            outputs = model(inputs, data['audio'])
            ipdb.set_trace()
            outputs['sal'][-1] = outputs['sal'][-1][0:curr_batch_size, :]
            cc_test = cc_score(outputs['sal'][-1], targets['salmap'], valid['sal'])
            nss_test = nss_score(outputs['sal'][-1], targets['binmap'], valid['sal'])
            sim_test = similarity(outputs['sal'][-1], targets['salmap'])

            auc_j_test = auc_judd(outputs['sal'][-1], targets['binmap'])
            auc_j.update(torch.mean(auc_j_test), nonzero(valid['sal'])[:, 0].size(0))

            if not opt.no_sigmoid_in_test:
                outputs['sal'] = torch.sigmoid(outputs['sal'][-1])

            if sum(valid['sal']) > 0:
                cc_tmp = cc_test / nonzero(valid['sal'])[:, 0].size(0)
                nss_tmp = nss_test / nonzero(valid['sal'])[:, 0].size(0)
                sim_tmp = sim_test / nonzero(valid['sal'])[:, 0].size(0)

                cc.update(cc_tmp, nonzero(valid['sal'])[:, 0].size(0))
                nss.update(nss_tmp, nonzero(valid['sal'])[:, 0].size(0))
                sim.update(sim_tmp, nonzero(valid['sal'])[:, 0].size(0))

            for j in range(outputs['sal'].size(0)):
                if not (i == 0 and j == 0) and targets['video'][j] != previous_video_id:
                    save_path = os.path.join(opt.result_path, opt.dataset, previous_video_id)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_video_results(output_buffer, save_path)
                    output_buffer = []
                output_buffer.append(outputs['sal'][j].data.cpu())
                previous_video_id = targets['video'][j]

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i+1) % 5 == 0 or (i+1) == len(data_loader):
                print('[{}/{}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CC {cc.val:.4f} ({cc.avg:.4f})\t'
                      'NSS {nss.val:.4f} ({nss.avg:.4f})\t'
                      'SIM {sim.val:.4f} ({sim.avg:.4f})\t'
                      'AUC {auc_j.val:.4f} ({auc_j.avg:.4f})'.format(
                       i + 1,
                       len(data_loader),
                       batch_time=batch_time,
                       cc=cc,
                       nss=nss,
                       sim=sim,
                       auc_j=auc_j))
    print('\n')
    save_path = os.path.join(opt.result_path, opt.dataset, previous_video_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_video_results(output_buffer, save_path)
