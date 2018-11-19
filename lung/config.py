# -- coding: utf-8 --
import numpy as np
from easydict import EasyDict as edict
from tools.data_preprocess import get_label_classes_from_xls

config = edict()
config.seqlen = 9
config.OLD_MODEL = True
config.FOR_DICOM = True
config.WINDOW_SHIFT = True
config.NORMALIZATION = True
config.IF_FLOAT = False

config.WINDOW_CENTER = -600
config.WINDOW_WIDTH = 1500

config.INPUT_IMAGE_SIZE = (512,512) # for loading boxes in the pascal_voc.py, should be okay for CT

# customized data loader, enabled in the train_end2end.py,s see anchor_loader.py and data_loader.py
########################
# CONFIG FOR DATA LOADER
########################

config.DATA_LOADER = edict()
config.DATA_LOADER.ENABLED = False
config.DATA_LOADER.NUM_WORKER = 3
config.DATA_LOADER.WORKER_TYPE = 'thread'

########################

## for dataset load function, support dicom and jpg, see io/image.py and dataset/pasco_voc.py gt_roidb func

# network related params
config.PIXEL_MEANS = np.array([0, 0, 0])
config.IMAGE_STRIDE = 0
config.RPN_FEAT_STRIDE = [4, 8, 16, 32]
config.RCNN_FEAT_STRIDE = [4, 8, 16, 32]
# config.FIXED_PARAMS = []
# config.FIXED_PARAMS_SHARED = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

# for multi_class loader or single class loader for training,see pasco_voc.py.
###########################
# CONFIG FOR CLASSIFICATION
###########################

config.CLASSES_LABELS_XLS_FILE_NAME = 'lung/classname_labelname_mapping_pjianwei.xls'
config.CLASSES, config.NODULE_CLASSES, config.CLASS_DICT, config.CONF_THRESH, config.CLASS_WEIGHTS, config.GT_CLASSES_WEIGHTS, \
        config.CLASS_Z_THRESHOLD_PRED, config.CLASS_Z_THRESHOLD_GT, config.GT_CLASS_Z_THRESHOLD_GT= get_label_classes_from_xls(config.CLASSES_LABELS_XLS_FILE_NAME)
config.NUM_CLASSES = len(config.CLASSES)

#########################

config.SCALES = [(800, 800)]  # first is scale (the shorter side); second is max size
config.ANCHOR_SCALES = [[32], [16], [8], [4]]

config.ANCHOR_RATIOS = (0.5, 1, 2)
config.NUM_ANCHORS = [len(anchor_scales) * len(config.ANCHOR_RATIOS) for anchor_scales in config.ANCHOR_SCALES]

#####################
# CONFIG FOR TRAINING
#####################

config.TRAIN = edict()

# # for module setting, related file: module.py
config.TRAIN.MULTI_THREAD_ENABLE = False
config.TRAIN.QUEUE_CAPACITY = 3


# # RPN anchor loader
# # rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 64


# # used for end2end training
# # RPN proposal
config.TRAIN.CXX_PROPOSAL = False   # for cuda proposal, by default: False

###################
# CONFIG FOR TEST
###################

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = False

config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 12000
config.TEST.RPN_POST_NMS_TOP_N = 1000
config.TEST.RPN_MIN_SIZE = [16] * 4

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 1000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.01

# same-box IOU threshold, used in post_process
config.TEST.OBJECT_COMPARE_THRESHOLD = np.array([1.6, 1.6])

# customized softmax threshold for model evaluator
# config.TEST.CONF_THRESHOLD = np.linspace(0.1, 0.85, num=16).tolist() + np.linspace(0.9, 0.975, num=4).tolist()\
#                            + np.linspace(0.99, 0.9975, num=4).tolist() + np.linspace(0.999, 0.99975, num=4).tolist()

config.TEST.CONF_THRESHOLD = np.linspace(0.1, 0.9, num=5).tolist()
#########################
# CONFIG FOR FIND_NODULES
#########################

config.FIND_NODULES = edict()

# 对于同一层面的不同类框，将其视为同一等价类的中心点偏移阈值/IOU，对于ground truth设为np.array([0., 0.]),仅适用于find_nodules_new:
config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED = np.array([1.6, 1.6])
config.FIND_NODULES.SAME_BOX_THRESHOLD_GT = np.array([0., 0.])

#同上，适用于find_nodules:
config.FIND_NODULES.SAME_BOX_THRESHOLD_PRED_OLD = 0.514
config.FIND_NODULES.SAME_BOX_THRESHOLD_GT_OLD = 1.0


# 对于不同层面两个框的匹配，将其视为二分图中一条边的中心点偏移阈值，对于ground truth一般应设置得更小
config.FIND_NODULES.SCORE_THRESHOLD_PRED = 0.6
config.FIND_NODULES.SCORE_THRESHOLD_GT = 0.4


# # 对于逐层匹配的贪心算法，我们每次只找前z_threshold个层面，对于ground truth一般应设置为1-2, 假设医生不标断层结节，则应该设置成１
# config.FIND_NODULES.Z_THRESHOLD_PRED = 3.
# config.FIND_NODULES.Z_THRESHOLD_GT = 3.

##########################
# CONFIG FOR OBJECT
##########################

config.ANCHOR = edict()

config.ANCHOR.CLASS_KEY = 'name'
config.ANCHOR.BNDBOX_KEY = 'bndbox'
config.ANCHOR.ADD_KW = ['prob']
config.ANCHOR.ADD_VALUE = [1.]
config.ANCHOR.KEY_LIST = ['name', 'Diameter', 'CT_value']
config.ANCHOR.BNDBOX_KEY_LIST = ['xmin', 'ymin', 'xmax', 'ymax']
config.ANCHOR.ALL_KEY_LIST = config.ANCHOR.BNDBOX_KEY_LIST + config.ANCHOR.ADD_KW + config.ANCHOR.KEY_LIST + ['sliceId']
config.ANCHOR.MATCHED_KEY_LIST = ['Bndbox List', 'Object Id', 'Pid', 'Type', 'SliceRange', 'Prob', 'Diameter', 'CT_value']

#######################

config.THICKNESS_THRESHOLD = 0
config.FSCORE_BETA = 1.0

#######################

config.MODEL_WEIGHT_LIST = [0.433, 0.440, 0.646, 0.504, 0.545, 0.556, 0.564]
config.MODEL_CONF_LIST = [0.433, 0.440, 0.646, 0.504, 0.545, 0.556, 0.564]
#config.MODEL_WEIGHT_LIST = [1., 1., 1., 1., 2., 2., 2.]
#config.MODEL_CONF_LIST = [1., 1., 1., 1., 1., 1., 1.]
config.MODEL_LIST = ['addcc300_9c_6class_1205-0016', 'add_dz_pretrain_dicom_9c_5class-0022', 'fpv3_ssd',
                     'ggn_review_test_9c_5class_0224-0018', 'hard0.3_6class_9c_pleural_lf-0020', 'lanf_9c_6class_pleural-0018',
                     'ssd-0020']
config.OBJ_FREQ_THRESH = 4
config.IF_ENSEMBLE = True
config.IF_STACKING = False
