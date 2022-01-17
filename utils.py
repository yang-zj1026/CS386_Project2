import csv
import numpy as np
import math
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def normalize_map(s_map, sim=False):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    if sim:
        norm_s_map /= np.sum(norm_s_map)
    return norm_s_map


def auc_judd_np(s_map, gt):
    # Normalize saliency map to have values between [0,1]
    s_map = normalize_map(s_map)
    gt = normalize_map(gt)

    S = s_map.ravel()
    F = gt.ravel()
    S_fix = S[F > 0]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)

    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio other saliency map values above threshold

    return np.trapz(tp, fp)


def auc_shuff(s_map, gt, other_map, n_splits=100, stepsize=0.1):
    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    s_map = normalize_map(s_map, sim=True)
    gt = normalize_map(gt, sim=True)
    return np.sum(np.minimum(s_map, gt))


def nss(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    x, y = np.where(gt == 1)
    temp = []
    for i in zip(x, y):
        temp.append(s_map_norm[i[0], i[1]])

    return np.mean(temp)


def cc(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    a = s_map_norm
    b = gt_norm
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


def get_logger(filename, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
