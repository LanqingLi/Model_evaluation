# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
from collections import OrderedDict

def get_cls_matrix(xlsx_path, thresh, category_keys=['Zero', 'Mild', 'Moderate', 'Severe']):
    xls = pd.ExcelFile(xlsx_path)
    eval_df1 = pd.read_excel(xls, 'multi-class_evaluation')
    eval_df1_thresh = eval_df1[eval_df1['threshold'] == thresh]
    eval_df1_thresh_filter1 = eval_df1_thresh[eval_df1_thresh['PatientID'] != 'average']
    eval_df1_thresh_filter2 = eval_df1_thresh_filter1[eval_df1_thresh_filter1['PatientID'] != 'total']

    #print eval_df1_thresh
    cls_matrix_df = pd.DataFrame(columns=category_keys)
    cls_matrix = np.zeros((len(category_keys), len(category_keys)))
    for index, row in eval_df1_thresh_filter2.iterrows():
        print row
        print row['CAC_risk_gt_filter']
        cls_matrix[category_keys.index(str(row['CAC_risk_gt_filter']))][category_keys.index(str(row['CAC_risk_pred_filter']))] += 1
    # for i in range(len(category_dict)):
    #     cls_row_df = pd.DataFrame(columns=category_dict.keys())
    #     for j in range(len(category_dict)):
    #         cls_row_df[category_dict.keys()[j]] = cls_matrix[i][j]
    #     print cls_row_df
    #     cls_matrix_df = cls_matrix_df.append(cls_row_df)
    cls_matrix_df = pd.DataFrame(cls_matrix, columns=category_keys, index=category_keys)
    print cls_matrix, cls_matrix_df


if __name__ == '__main__':
    xlsx_path = '/mnt/data2/model_evaluation/examples/heart/OnlineIter/HeartSemanticSegEvaluation_result/seqlen3_60_test_set/multi_class_evaluation.xlsx'
    thresh = 0.5
    get_cls_matrix(xlsx_path, thresh)
