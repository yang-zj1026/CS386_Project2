import numpy as np
import torch
import torch.nn.functional as F
import ipdb
import time


def logit(x):
    return np.log(x / (1 - x + 1e-08) + 1e-08)


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def cc_score(x, y, weights, batch_average=False, reduce=True):
    x = x.squeeze(1)
    x = torch.sigmoid(x)
    y = y.squeeze(1)
    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    mean_y = torch.mean(torch.mean(y, 1, keepdim=True), 2, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(torch.sum(torch.mul(xm, ym), 1, keepdim=True), 2, keepdim=True)
    r_den_x = torch.sum(torch.sum(torch.mul(xm, xm), 1, keepdim=True), 2, keepdim=True)
    r_den_y = torch.sum(torch.sum(torch.mul(ym, ym), 1, keepdim=True), 2, keepdim=True) + np.asscalar(
        np.finfo(np.float32).eps)
    r_val = torch.div(r_num, torch.sqrt(torch.mul(r_den_x, r_den_y)))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = torch.sum(r_val)
        else:
            r_val = r_val
    return r_val


def nss_score(x, y, weights, batch_average=False, reduce=True):
    x = x.squeeze(1)
    x = torch.sigmoid(x)
    y = y.squeeze(1)
    y = torch.gt(y, 0.0).float()

    mean_x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    std_x = torch.sqrt(torch.mean(torch.mean(torch.pow(torch.sub(x, mean_x), 2), 1, keepdim=True), 2, keepdim=True))
    x_norm = torch.div(torch.sub(x, mean_x), std_x)
    r_num = torch.sum(torch.sum(torch.mul(x_norm, y), 1, keepdim=True), 2, keepdim=True)
    r_den = torch.sum(torch.sum(y, 1, keepdim=True), 2, keepdim=True)
    r_val = torch.div(r_num, r_den + np.asscalar(np.finfo(np.float32).eps))
    r_val = torch.mul(r_val.squeeze(), weights)
    if batch_average:
        r_val = torch.sum(r_val) / torch.sum(weights)
    else:
        if reduce:
            r_val = torch.sum(r_val)
        else:
            r_val = r_val
    return r_val


def batch_image_sum(x):
    x = torch.sum(torch.sum(x, 1, keepdim=True), 2, keepdim=True)
    return x


def batch_image_mean(x):
    x = torch.mean(torch.mean(x, 1, keepdim=True), 2, keepdim=True)
    return x


def cross_entropy_loss(output, label, weights, batch_average=False, reduce=True):
    batch_size = output.size(0)
    output = output.view(batch_size, -1)
    label = label.view(batch_size, -1)

    label = label / 255
    final_loss = F.binary_cross_entropy_with_logits(output, label, reduction='none').sum(1)
    final_loss = final_loss * weights

    if reduce:
        final_loss = torch.sum(final_loss)
    if batch_average:
        final_loss /= torch.sum(weights)

    return final_loss


def normalize_map(s_map, sim=False):
    # normalize the salience map
    batch_size, height, width = s_map.shape
    s_map = s_map.view(s_map.size(0), -1)
    s_map -= s_map.min(1, keepdim=True)[0]
    s_map /= s_map.max(1, keepdim=True)[0]

    if sim:
        s_map /= s_map.sum(1, keepdim=True)

    norm_s_map = s_map.view(batch_size, height, width)
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = s_map.squeeze(1)
    s_map = torch.sigmoid(s_map)
    gt = gt.squeeze(1)

    s_map = (s_map - torch.min(s_map)) / (s_map.max() - s_map.min())
    gt = (gt - gt.min()) / (gt.max() - gt.min())
    assert torch.max(gt) == 1.0, \
        'Ground truth not discretized properly max value > 1.0'
    assert torch.max(s_map) == 1.0, \
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    zeros = torch.zeros_like(gt)
    thresholds = torch.where(gt > 0, s_map, zeros).view(gt.shape[0], -1)
    s_map = s_map.view(s_map.shape[0], -1)

    # num fixations is no. of salience map values at gt >0
    num_fixations = torch.sum(thresholds > 0, dim=1)
    n_fix_max = torch.max(num_fixations).item()
    num_pixels = gt.shape[1] * gt.shape[2]

    thresholds = torch.sort(thresholds, descending=True)[0][:, :n_fix_max]
    s_map = s_map.permute(1, 0)
    thresholds = thresholds.permute(1, 0)

    tp = torch.full((n_fix_max + 2, gt.shape[0]), 1.).to(gt.device)
    fp = torch.full((n_fix_max + 2, gt.shape[0]), 1.).to(gt.device)
    tp[0], fp[0] = 0., 0.
    # st = time.time()
    for k, thresh in enumerate(thresholds):
        # Total number of saliency map values above threshold
        above_th = torch.sum(s_map >= thresh, dim=0, keepdim=True)
        # Ratio saliency map values at fixation locations above threshold
        tp[k + 1] = torch.where(thresh > 0, (k + 1) / num_fixations, tp[k + 1])
        # Ratio other saliency map values above threshold
        fp[k + 1] = torch.where(thresh > 0, (above_th - k - 1) / (num_pixels - num_fixations), fp[k + 1])

    tp, fp = tp.permute(1, 0), fp.permute(1, 0)
    res = torch.trapz(tp, fp)
    # print('Time:', time.time() - st)
    return res


def auc_shuff_np(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
    other_map = (other_map - np.min(other_map)) / (np.max(other_map) - np.min(other_map))

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map
    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((n_splits, Nfixations_oth), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i, thresh in enumerate(allthreshes):
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth
        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    s_map = s_map.squeeze(1)
    s_map = torch.sigmoid(s_map)
    gt = gt.squeeze(1)

    s_map = normalize_map(s_map, sim=True)
    gt = normalize_map(gt, sim=True)
    return torch.sum(torch.min(s_map, gt))
