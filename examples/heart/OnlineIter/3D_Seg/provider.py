import os
import numpy as np
# from tools.crf import CRF_process
import mxnet as mx
import sys

sys.path.append("..")
from data_iter import FileIter
# from symbol import symbol_deep_dense
from config import config
# from tools.epoch_end_call import regex_resume
import time
# from tools.contour_draw import contour_and_draw_doc, contour_and_draw
# from preprocessing.window_converter import window_convert
# import matplotlib.pyplot as plt
# from  symbol.symbol_deep_dense  import get_symbol_valid
class Predict(object):
    def __init__(self, model_prefix=os.path.join(config.model_dir, config.test.model_name),
                 epoch=config.test.epoch, gpus=config.gpu, num_classes=config.num_cls,
                 model_dir=config.model_dir, data_name='data', label_name='softmax_label'):
        self.model_prefix = model_prefix
        self.epoch = epoch
        self.ctx = [mx.gpu(i) for i in gpus]
        self.num_classes = num_classes

        self.batch_size = config.test.batch_size
        self.mod = None
        self.data_name = data_name
        self.model_dir = model_dir
        self.label_name = label_name
        self.seqlen = config.seqlen
        self.img_shape = config.img_shape
        self.set_mod()

    def set_mod(self):
        '''
        :param model_prefix: type string. model prefix include the dir
        :param epoch: type int. define which epoch to be loaded
        :param gpus: list of available gpus, e.g [0,1]
        :return: mx.mod.module  for predicting
        '''

        symbol, args, auxs = mx.model.load_checkpoint(self.model_prefix, self.epoch)
        print 'get mod', self.model_prefix, ' epoch ', self.epoch
        # symbol=get_symbol_valid()
        self.mod = mx.mod.Module(symbol=symbol, context=self.ctx, label_names=None)
        self.mod.bind(data_shapes=[('data', (len(self.ctx), self.seqlen, self.img_shape[0], self.img_shape[1]))],for_training=False)
        self.mod.set_params(args, auxs, allow_missing=True)

    def predict(self, ct_sequence):
        data_iter = mx.io.NDArrayIter(data=ct_sequence, batch_size=self.batch_size)
        input_shape = ct_sequence.shape
        pred_lbl = np.zeros((input_shape[0], config.num_cls, input_shape[2], input_shape[3]))
        pred_lbl_crf = np.zeros((input_shape[0], input_shape[2], input_shape[3]))

        for t, batch in enumerate(data_iter):
            self.mod.forward(batch, is_train=True)
            output = self.mod.get_outputs()[0].asnumpy()
            print output.shape
            ######Debug to show the result of predict#######
            # imgs = batch.data[0].asnumpy()
            # print output.shape
            # for idx in range(imgs.shape[0]):
            #     img = imgs[idx].transpose(1, 2, 0)
            #     img = window_convert(img, 40, 80)
            #     lbl = np.argmax(output, axis=1)[idx]
            #     print lbl.shape
            #     # if np.sum(lbl[idx]) > 0:
            #     img, _ = contour_and_draw(img, lbl)
            #     plt.imshow(img)
            #     plt.show()
            ##############
            origin = batch.data[0].asnumpy()
            for i in range(output.shape[0]):
                if not t * self.batch_size + i < pred_lbl.shape[0]: continue
                prob_img = output[i]
                origin_img = origin[i, 0, :, :]
                pred_lbl[t * self.batch_size + i] = prob_img
                # new_img = CRF_process(prob_img, origin_img, M=config.num_cls)
                # pred_lbl_crf[t * self.batch_size + i] = new_img
                # pred_lbl_crf1 = pred_lbl_crf[:, np.newaxis, :, :]
                # pred_lbl_crf1 = np.repeat(pred_lbl_crf1, repeats=2, axis=1)
                # print 'pred_lbl_crf', pred_lbl_crf.shape, pred_lbl.shape
        # return pred_lbl > 0.5
        return pred_lbl

# def main():
#     predictor = Predict()
#     eval_data = FileIter(
#             root_dir=config.valid_dir,
#             persons=config.valid_person,
#             rgb_mean=(0, 0, 0),
#             )
#     for data in eval_data:
#         start = time.time()
#         if data is None:
#             break
#         if data['pid'] is None: continue
#         # no crf
#         target_dir = config.validating.pred_lmap_dir+'_no_crf'
#         if not os.path.exists(target_dir):
#             os.mkdir(target_dir)
#         # print data['pid']
#         target_file = os.path.join(target_dir, data['pid']+'.npy')
#
#         # crf
#         target_dir_crf = config.validating.pred_lmap_dir+'_crf'
#         if not os.path.exists(target_dir_crf):
#             os.mkdir(target_dir_crf)
#         target_file_crf = os.path.join(target_dir_crf, data['pid'] + '.npy')
#         # crf
#
#         if os.path.exists(target_file) and os.path.exists(target_file_crf): continue
#
#         pred_lbl, pred_lbl_crf = predictor.predict(data)
#         print pred_lbl.shape, pred_lbl_crf.shape
#         if not os.path.exists(target_file):
#             np.save(target_file, pred_lbl)
#         if not os.path.exists(target_file_crf):
#             np.save(target_file_crf, pred_lbl_crf)
#         print time.time()-start
#
# if __name__ == "__main__":
#     main()
