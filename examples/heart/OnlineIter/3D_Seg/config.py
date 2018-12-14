# --coding:utf-8--
from easydict import EasyDict as edict
import os
import numpy as np


config = edict()
config.num_cls = 3
config.gpu = [0, 1, 2, 3]

# config.TEST = edict()
#
#config.TEST.CONF_THRESHOLD = np.linspace(0.001, 0.009, num=9).tolist() + np.linspace(0.01, 0.09, num=9).tolist() + np.linspace(0.1, 0.9, num=9).tolist()
#
# config.CLASSES_LABELS_XLS_FILE_NAME = 'brain/classname_labelname_mapping.xls'
# config.CLASSES, config.SEG_CLASSES, config.CLASS_DICT, config.CONF_THRESH, config.CLASS_WEIGHTS = get_label_classes_from_xls_seg(config.CLASSES_LABELS_XLS_FILE_NAME)
# config.NUM_CLASSES = len(config.CLASSES)

# config.FSCORE_BETA = 1.

# binary classification threshold for drawing contour plot for single threshold for comparison
config.THRESH = 0.5

config.model_dir = '/mnt/data2/calcium_scoring/CAC-Scoring-Train/model'


config.test = edict()
config.test.batch_size = 4
config.test.model_name = 'deep_dense_heart_3d_recon_bstrap_randcrop_new_seqlen9_weight3_second_batch'
config.test.epoch = 6
config.test.test_dir= '/media/tx-eva-cc/data/3D_recon_data/test_data/test_data/FW_second_batch_test_labeled'
config.test.test_person = os.listdir(config.test.test_dir)

config.img_shape = (512, 512)

config.window_width = 800
config.window_center = 300

# config.min_crop_shape = [256, 256]
# config.tgt_crop_shape = [256, 256]
# config.fix_crop_shape = False
# config.crop_dof = 1

config.seqlen = 9
