#!/home/lanruo_wan/venv python
#_*_coding:utf-8_*_
import numpy as np
import sys, os
from mxnet.io import DataIter
import numpy.random as npr
import matplotlib.pyplot as plt
# from preprocessing.window_converter import window_convert
# from tools.contour_draw import contour_and_draw
import nrrd
import cv2
from config import config
# from preprocessing import window_converter
# from preprocessing.off_bone import get_off_bone
# from tools.conversion_img import get_imgs

def get_off_bone(data, thresh):
    # data should be nrrd array
    indexes = np.where(data > thresh)
    #  -1024
    data[indexes] = -1024
    return data


def get_off_negative(data, thresh):
    # data should be nrrd array
    indexes = np.where(data < thresh)
    data[indexes] = thresh
    return data


class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """

    def __init__(self, root_dir, persons=['zhangyu', 'zhaozhao', 'weiren', 'wuchunxue', 'liyanong'],
                 rgb_mean=(0, 0, 0),
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=2,
                 is_train=True,
                 negative_mining_ratio=1):
        super(FileIter, self).__init__()
        self.cursor = -1
        #  tile or other
        self.mode = 'tile'
        self.negative_mining_ratio = negative_mining_ratio
        self.is_train = is_train
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.persons = persons
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.data_name = data_name
        self.label_name = label_name
        self.pids = self.get_training_path()
        npr.shuffle(self.pids)
        self.num = len(self.pids)
        self.sid = None
        print 'total ct ', self.num
        self.data, self.label, _ = self._read()

    def get_training_path(self):
        sid_label_list = []

        for person in self.persons:
            ppath = os.path.join(self.root_dir, person)
            assert os.path.exists(ppath), 'person path not exist: ' + ppath
            phases = os.listdir(ppath)
            for phase in phases:
                cpath = os.path.join(ppath, phase)
                dpath = os.path.join(cpath, 'data')
                lpath = os.path.join(cpath, 'label')
                assert os.path.exists(lpath), ' label dir not exist under: ' + cpath

                sids = os.listdir(lpath)
                for sid in sids:
                    sid_path = os.path.join(lpath, sid)
                    items = os.listdir(sid_path)
                    if len(items) >= 2:
                        suffix_check = [item.split('.')[-1] for item in items]
                        if suffix_check.count('nrrd') == 2:
                            sid_label_list.append(sid_path)
        np.random.shuffle(sid_label_list)
        # sids = os.listdir(self.root_dir)
        # for sid in sids:
        #     sid_path = os.path.join(self.root_dir, sid)
        #     items = os.listdir(sid_path)
        #     if len(items) >= 2:
        #         suffix_check = [item.split('.')[-1] for item in items]
        #         if suffix_check.count('nrrd') == 2:
        #             sid_label_list.append(sid_path)
        return sid_label_list

    def _read(self):
        sid_path = self.pids[self.cursor]
        self.sid = sid_path
        data, label, raw = self._read_sid(sid_path)
        return data, label, raw

    def label_convert(self, lbl):
        nlbl = np.array(lbl > 0, dtype=np.int16)
        return nlbl

    def label_convert_brainRegion(self, lbl):
        nlbl = np.array(lbl, dtype=np.int16)
        return nlbl

    def _read_sid(self, sid_path):
        # start = time.time()
        items = os.listdir(sid_path)
        print '@@@@@@@@@@@@sid_path', sid_path
        img_nrrd = [item for item in items if item.endswith('.nrrd') and not item.endswith('abel.nrrd')]
        lbl_nrrd = [item for item in items if item.endswith('abel.nrrd')]
        assert len(img_nrrd) == 1 and len(lbl_nrrd) == 1, 'img or lbl nrrd not the only one'
        img_nrrd = img_nrrd[0]
        lbl_nrrd = lbl_nrrd[0]

        try:
            t = nrrd.read(os.path.join(sid_path, img_nrrd))
            t_lbl = nrrd.read(os.path.join(sid_path, lbl_nrrd))
            t_lbl_new = []
            t_lbl_new.append(t_lbl[0])
            t_lbl_new.append(t_lbl[1])
            print 't', t[0].shape, t_lbl[0].shape

            tmp_dir = os.path.join(sid_path, img_nrrd)
            # print tmp_dir.split('/')[8]
            if tmp_dir.split('/')[8] == 'naoshi':
                # print "t[0]",t[0].shape
                t0 = t[0].transpose(1,0,2)
                t_lbl0 = t_lbl[0].transpose(1,0,2)
                # print 't[]',t0.shape
            else:
                t0 = t[0]
                t_lbl0 = t_lbl[0]
            imgs = t0  # 由于对数据有去骨和不去骨的预处理，为了便于后面显示画图，所以有imgs(有可能经过预处理)，raws（未经过处理的原始数据）
            raws = imgs.copy().transpose(2, 0, 1)  # N*512*512
            lbls = t_lbl0
            print 'imgs-t0', imgs.shape, lbls.shape


            del t
            del t_lbl
        except Exception as e:
            print e
            return None, None, None

        lbls = self.label_convert(lbls)  # 512 512 N
        # lbls = self.label_convert_brainRegion(lbls)  # 512 512 N
        assert imgs.shape[-1] > 2
        assert lbls.shape[-1] > 2

        if config.is_getOffBone:
            imgs = get_off_bone(imgs, thresh=150)
        # imgs = get_off_negative(imgs, thresh=-200)

        #########
        # imgs = window_converter.preProsessing_CT(imgs)

        if self.mode == 'tile':
            imgs = imgs.reshape((512, 512, 1, -1))
            imgs = np.tile(imgs, (1, 1, 3, 1))  # 512,512,3,N
            imgs = np.transpose(imgs, (3, 2, 0, 1))  # N,3,512,512
        else:
            image = [imgs[:, :, i:i + 3] for i in range(imgs.shape[-1] - 2)]
            imgs = np.array(image).transpose((0, 3, 1, 2))
            lbls = lbls[:, :, 1:lbls.shape[-1] - 1]

        reshaped_mean = self.mean.reshape(1, 3, 1, 1)
        imgs = imgs - reshaped_mean  # imgs (N, 3, 512, 512)
        raws = np.repeat(raws[:, np.newaxis, :, :], axis=1, repeats=3)  # N*3*512*512

        lbls = np.transpose(lbls, (2, 0, 1))  # N 512 512

        print 'reshape', imgs.shape, lbls.shape, raws.shape




        # Positive and negative sample balance
        if self.is_train:
            print 'is train'
            lblss = []
            imgss = []
            rawss = []
            keep = []
            count = 0
            i = 0
            for lbl, img, raw in zip(lbls, imgs, raws):
                if np.max(lbl) > 0:
                    lblss.append(lbl)
                    imgss.append(img)
                    rawss.append(raw)
                    count += 1
                else:
                    keep.append(i)
                i += 1
            if lbls.shape[0] - (self.negative_mining_ratio + 1) * count <= 0: #若一个sid中，正样本的数量多余负样本数量，则不作样本平衡
                lblss = lbls
                imgss = imgs
                rawss = raws
            else:
                bg_inds = npr.choice(keep, size=self.negative_mining_ratio * count, replace=False) #若正样本数量少于负样本，则随机选择出和正样本同样数量的负样本
                lbl = lbls[bg_inds]
                img = imgs[bg_inds]
                raw = raws[bg_inds]
                for label, image, raww in zip(lbl, img, raw):
                    lblss.append(label)
                    imgss.append(image)
                    rawss.append(raww)
            lblss = np.array(lblss)
            imgss = np.array(imgss)
            rawss = np.array(rawss)
            # 若正样本数为0且negative_mining_ratio=1,前面的样本平衡中增加的负样本也为0, 也就是说该sid是正常人，没有出血，
            # 为了保持数据迭代器输出的一致性，选取第一张负样本图像，后面的if将其复制成batch_size份用于输出
            if len(lblss) == 0 and len(imgss) == 0:
                print sid_path
                lblss = lbls[0, :, :]
                imgss = imgs[0, :, :, :]
                rawss = raws[0, :, :, :]

                lblss = lblss[np.newaxis, :, :]
                imgss = imgss[np.newaxis, :, :, :]
                rawss = rawss[np.newaxis, :, :, :]
            if lblss.shape[0] == 1 and imgss.shape[0] == 1:
                lblss = np.tile(lblss, (self.batch_size, 1, 1))
                imgss = np.tile(imgss, (self.batch_size, 1, 1, 1))
                rawss = np.tile(rawss, (self.batch_size, 1, 1, 1))

            index = [i for i in range(lblss.shape[0])]
            npr.shuffle(index)
            lbls = lblss[index]
            imgs = imgss[index]
            raws = rawss[index]

        # if lbls.shape[0] == 1 and imgs.shape[0] == 1:  #防止图像数目小于batch_size
        #     lbls = np.tile(lbls, (self.batch_size, 1, 1)) #比如 a = np.array([0,1,2]),    np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍，即最终得到
        #     imgs = np.tile(imgs, (self.batch_size, 1, 1, 1))
        #     raws = np.tile(raws, (self.batch_size, 1, 1, 1))

        #print 'imgs.shape in _read_sid func: ', self.sid,imgs.shape[0], imgs.shape
        assert imgs.shape[0] >= 2
        assert lbls.shape[0] >= 2
        assert raws.shape[0] >= 2
        ###################################
        ###############Debug##########################
        # for image, label in zip(imgs, lbls):
        #     image = image[1, :, :]
        #     image = np.tile(image, (3, 1, 1))
        #     image = image.transpose((1,2,0))
        #     image = window_convert(image, 40, 80)
        #     image, _ = contour_and_draw(image, label)
        #     plt.imshow(image)
        #     plt.show()
        # print imgs.shape, lbls.shape
        ##################################
        assert imgs.shape[0] == lbls.shape[0]
        assert imgs.shape[2] == lbls.shape[1]
        assert imgs.shape[3] == lbls.shape[2]

        # print 'read_sid time ', time.time()-start

        # lals_tran = np.repeat(lbls[:, np.newaxis, :, :], repeats=2, axis=1)
        # imgs, lals_tran, raws = get_imgs(imgs, lals_tran, raws)
        # lbls = lals_tran[:, 1, :, :]
        lbls = lbls.transpose((1, 2, 0))
        print 'return', imgs.shape, lbls.shape, raws.shape
        t_lbl_new[0] = lbls
        return imgs, t_lbl_new, raws

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(self.data_name, self.data.shape)]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(self.label_name, self.label.shape)]

    def get_batch_size(self):
        return 1

    def reset(self):
        self.cursor = -1

    def iter_next(self):
        self.cursor += 1
        if (self.cursor <= self.num - 1):
            # if (self.cursor < 100):
            return True
        else:
            return False


    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label, raw = self._read()
            #print '%%%%%%%%%%%%%%%%', type(self.data),np.shape(self.data)
            return {self.data_name: self.data,
                    self.label_name: self.label,
                    'pid': self.pids[self.cursor].split('/')[-1],
                    'raw': raw
                    }
        else:
            return None
