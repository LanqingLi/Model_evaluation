# -*-coding:utf-8-*-
import os, shutil

def mv_img(src_dir, tgt_dir, img_name):
    for file_name in os.listdir(src_dir):
        file_dir = os.path.join(src_dir, file_name)
        shutil.copy(os.path.join(file_dir, img_name), os.path.join(tgt_dir, file_name, img_name))



if __name__ == '__main__':
    tgt_dir = '/mnt/data2/model_evaluation/examples/heart/OnlineIter/3D_Seg/HeartSemanticSegEvaluation_mask'
    src_dir = '/media/tx-eva-cc/data/3D_recon_data/test_data/test_data/FW_second_batch_test/FW_second_batch_test/FW_second_batch_test'
    img_name = 'img.nrrd'
    mv_img(src_dir, tgt_dir, img_name)
