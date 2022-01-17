import os
import argparse
import numpy as np
import ipdb
import scipy.io as sio
from utils import AverageMeter, cc, similarity, nss, auc_judd_np, auc_shuff, get_logger
from datasets.saliency_db import pil_loader_sal, read_sal_text
from PIL import Image


def evaluate(dataset, salmap_path, pred_path, logger):
    CC = AverageMeter()
    NSS = AverageMeter()
    SIM = AverageMeter()
    AUC_J = AverageMeter()
    S_AUC = AverageMeter()
    salmap_path = os.path.join(salmap_path, dataset)

    logger.info('--------------- Dataset: {} ---------------'.format(dataset))
    for split in range(1, 4):
        annotation_path = './data/fold_lists/{}_list_test_{}_fps.txt'.format(dataset, split)
        pred_res_path = os.path.join(pred_path, 'split{}_results'.format(split), dataset)
        data = read_sal_text(annotation_path)
        video_names = data['names']
        video_nframes = data['nframes']

        for i in range(len(video_names)):
            annot_path = os.path.join(salmap_path, video_names[i], 'maps')
            annot_path_bin = os.path.join(salmap_path, video_names[i])
            pred_video_path = os.path.join(pred_res_path, video_names[i])
            file_list = get_pred_files(pred_video_path)

            n_frames = int(video_nframes[i])
            if n_frames <= 1:
                continue

            eyeMap_all = []
            for j in range(n_frames):
                tmp_mat = sio.loadmat(os.path.join(annot_path_bin, 'fixMap_{:05d}.mat'.format(j + 1)))
                eyeMap = np.array(Image.fromarray(tmp_mat['eyeMap']))
                eyeMap_all.append(eyeMap)
            shuffle_map = get_shuffle_map(eyeMap_all)

            cc_video = []
            nss_video = []
            sim_video = []
            auc_j_video = []
            s_auc_video = []
            for k in range(len(file_list)):
                tmp = file_list[k].split('.')
                tmp = tmp[0].split('_')[-1]
                frame_num = int(tmp)
                frame_path = os.path.join(pred_video_path, file_list[k])
                if frame_num <= n_frames:
                    # print('video %d of %d: frame %d of %d' % (i+1, len(video_names), k+1, len(file_list)))
                    pred_sal = np.array(pil_loader_sal(frame_path, new_size=(640, 480))).astype(np.float64)
                    gt_sal_path = os.path.join(annot_path, 'eyeMap_{:05d}.jpg'.format(frame_num))
                    gt_salmap = np.array(pil_loader_sal(gt_sal_path, resize=False)).astype(np.float64)
                    gt_fixmap = eyeMap_all[frame_num-1].astype(np.float64)
                    shuffle_map1 = shuffle_map.astype(np.float64)
                    shuffle_map1[gt_fixmap == 1] = 0

                    cc_video.append(cc(pred_sal, gt_salmap))
                    nss_video.append(nss(pred_sal, gt_fixmap))
                    sim_video.append(similarity(pred_sal, gt_salmap))
                    auc_j_video.append(auc_judd_np(pred_sal, gt_fixmap))
                    s_auc_video.append(auc_shuff(pred_sal, gt_fixmap, shuffle_map1))

            video_len = len(nss_video)
            cc_test = np.mean(cc_video)
            nss_test = np.mean(nss_video)
            sim_test = np.mean(sim_video)
            auc_j_test = np.mean(auc_j_video)
            s_auc_test = np.mean(s_auc_video)

            CC.update(cc_test, video_len)
            NSS.update(nss_test, video_len)
            SIM.update(sim_test, video_len)
            AUC_J.update(auc_j_test, video_len)
            S_AUC.update(s_auc_test, video_len)

            logger.info('Video: {:20s}\t CC {:.4f}\t NSS {:.4f}\t SIM {:.4f}\t AUC_J {:.4f}\t S_AUC {:.4f}'.format(
                   video_names[i], np.mean(cc_video), np.mean(nss_video),
                   np.mean(sim_video), np.mean(auc_j_video),
                   np.mean(s_auc_video)))

    logger.info('Overall: {:20s}\t CC {:.4f}\t NSS {:.4f}\t SIM {:.4f}\t AUC_J {:.4f}\t sAUC {:.4f}\n'.format(
          dataset, CC.avg, NSS.avg, SIM.avg, AUC_J.avg, S_AUC.avg))


def get_pred_files(rootdir):
    file_list = []
    for path, d, files in os.walk(rootdir):
        for file in files:
            if file.endswith('jpg'):
                file_list.append(file)
    return file_list


def get_shuffle_map(eyemap_all):
    shuffle_map = np.zeros_like(eyemap_all[0])
    for eyeMap in eyemap_all:
        shuffle_map += eyeMap

    return shuffle_map > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STAViS options and parameters')

    # Results path
    parser.add_argument(
        '--result_dir',
        default='./experiments/visual_train_test',
        type=str,
        help='Directory path of STAViS experiments')
    args = parser.parse_args()

    datasets = ['AVAD', 'Coutrot_db1', 'Coutrot_db2']
    logger = get_logger(os.path.join(args.result_dir, 'test_result.log'))

    for dataset in datasets:
        evaluate(dataset, 'data/annotations', args.result_dir, logger)
