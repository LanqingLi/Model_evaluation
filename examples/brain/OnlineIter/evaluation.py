#!/home/lanruo_wan/venv python
#_*_coding:utf-8_*_

import sys
import os

sys.path.append('..')
import cv2
import mxnet as mx
from config import config
from data_brain import FileIter
from provider import Predict
import numpy as np
# from preprocessing.window_converter import window_convert
# from tools.contour_draw import contour_and_draw
# from metric.custom_metric import Ignore_BG_Accuracy_Metric, DICEMultiMetric, CRFMetric,PRiDICEMultiMetric, PRiDICEMultiMetric2
# from tools.postprocessing import postProcessing
import matplotlib.pyplot as plt
from model_eval.brain.evaluator import BrainSemanticSegEvaluatorOnlineIter
eval_data = FileIter(
    root_dir=config.validating.valid_dir,
    persons=config.validating.valid_person,
    is_train=False,
    rgb_mean=(0, 0, 0)
)

predictor = Predict()
count = 0

# defined the metric
# dice_metric = PRiDICEMultiMetric2()
# dice_metric.reset()

overlap_val = 0
sum_val = 0
fp_val = 0
fn_val = 0

overlap_sid = 0
sum_sid = 0
fp_sid = 0
fn_sid = 0
num_manualLabelSid = 0
num_preLabelSid = 0

num_manualLabel = 0
num_preLabel = 0

# fname = config.validating.fname
# fobj = open(fname, 'a+')  # 这里的a意思是追加，这样在加了之后就不会覆盖掉源文件中的内容，如果是w则会覆盖。

shape_data = 256

cls_label_xls_path = '/mnt/data2/model_evaluation/example/brain/OnlineIter/classname_labelname_mapping.xls'

#一次只能生成multi_class_evaluation和binary_class_evaluation中的一个,除非每次重新初始化BrainSemanticSegEvaluatorOnlineIter
#并修改生成文件名xlsx_name和json_name
brain_eval = BrainSemanticSegEvaluatorOnlineIter(cls_label_xls_path=cls_label_xls_path,
                                                 data_iter=eval_data,
                                                 predictor=predictor.predict,
                                                 predict_key='data',
                                                 gt_key='softmax_label',
                                                 img_key='raw',
                                                 patient_key='pid',
                                                 conf_thresh=[0.1, 0.9],
                                                 if_save_mask=True
                                                 )



# 画单阈值contour
brain_eval.binary_class_contour_plot_single_thresh()
#print os.getcwd()


# # 画多阈值contour
brain_eval.binary_class_contour_plot_multi_thresh()
#
# # 二分类统计指标
brain_eval = BrainSemanticSegEvaluatorOnlineIter(cls_label_xls_path=cls_label_xls_path,
                                                 data_iter=eval_data,
                                                 predictor=predictor.predict,
                                                 predict_key='data',
                                                 gt_key='softmax_label',
                                                 img_key='raw',
                                                 patient_key='pid',
                                                 xlsx_name='binary_class_evaluation.xlsx',
                                                 json_name='binary_class_evaluation',
                                                 conf_thresh=[0.1, 0.9],
                                                 if_save_mask=False
                                                 )
brain_eval.binary_class_evaluation_light()
#
# #多分类统计指标
brain_eval = BrainSemanticSegEvaluatorOnlineIter(cls_label_xls_path=cls_label_xls_path,
                                                 data_iter=eval_data,
                                                 predictor=predictor.predict,
                                                 predict_key='data',
                                                 gt_key='softmax_label',
                                                 img_key='raw',
                                                 patient_key='pid',
                                                 xlsx_name='multi_class_evaluation.xlsx',
                                                 json_name='multi_class_evaluation',
                                                 conf_thresh=[0.1, 0.9],
                                                 if_save_mask=False
                                                 )
brain_eval.multi_class_evaluation_light()
