from model_eval.tools.plot import RP_plot_xlsx, RP_plot_json, RP_plot_json_tot, RP_plot_xlsx_tot
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Infervision auto test')
    parser.add_argument('--json',
                        help='whether to generate RP plot from .json',
                        action='store_true')
    parser.add_argument('--AUC',
                        help='whether to calculate AUC of the curve',
                        action='store_true')
    parser.add_argument('--total',
                        help='whether to draw RP for each patient or only total stat',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # xlsx_save_dir = '/mnt/data2/model_evaluation/LungNoduleEvaluation_result'
    # xlsx_name = 'LungNoduleEvaluation.xlsx'
    # json_save_dir = '/mnt/data2/model_evaluation/LungNoduleEvaluation_result'
    # json_name = 'LungNoduleEvaluation_multi-class_evaluation.json'
    # sheet_name = 'binary-class_evaluation'
    xlsx_save_dir = '/mnt/data2/model_evaluation_dev/BrainSemanticSegEvaluation_result'
    xlsx_name = 'BrainSemanticSegEvaluation.xlsx'
    json_save_dir = '/mnt/data2/model_evaluation_dev/BrainSemanticSegEvaluation_result'
    json_name = 'BrainSemanticSegEvaluation_multi-class_evaluation.json'
    sheet_name = 'multi-class_evaluation'
    xmin = 0.
    xmax = 1.
    cls_key = 'PatientID'
    args = parse_args()
    if args.total:
        if args.json:
            RP_plot_json_tot(json_save_dir=json_save_dir, json_name=json_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax,
                         cls_key=cls_key)
        else:
            RP_plot_xlsx_tot(xlsx_save_dir=xlsx_save_dir, xlsx_name=xlsx_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax,
                         cls_key=cls_key)
    else:
        if args.json:
            RP_plot_json(json_save_dir=json_save_dir, json_name=json_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax, cls_key=cls_key)
        else:
            RP_plot_xlsx(xlsx_save_dir=xlsx_save_dir, xlsx_name=xlsx_name, sheet_name=sheet_name, xmin=xmin, xmax=xmax, cls_key=cls_key)