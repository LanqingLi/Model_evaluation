from easydict import EasyDict as edict
import os
import numpy as np


config = edict()
config.num_cls = 2
config.gpu = [0, 1]
config.is_getOffBone = True
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

config.model_dir = '/mnt/data2/model_evaluation/examples/model'


config.validating = edict()
config.validating.batch_size = 2
config.validating.model_name = 'deep_dense_changzheng12+WHTJ+DLZS_bstrap_atrous_randcrop_new_seqlen9'
config.validating.epoch = 111
config.validating.valid_dir='/mnt/data2/model_evaluation/examples/data/test_data1'
config.validating.valid_person = os.listdir(config.validating.valid_dir)
