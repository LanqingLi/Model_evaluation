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
from window_converter import window_convert
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
        the list file of image and label, every line owns the form:
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
                 negative_mining_ratio=1,
                 if_binary=True):
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
        self.if_binary = if_binary
        self.img_shape = config.img_shape
	self.window_center = config.window_center
	self.window_width = config.window_width
        self.data, self.label, _, self.voxel_vol = self._read()
	



    def get_training_path(self):
        '''
        返回打乱顺序的病人列表
        '''
        sid_label_list = []

        for person in self.persons:
            ppath = os.path.join(self.root_dir, person)
            assert os.path.exists(ppath), 'person path not exist: ' + ppath
            phases = os.listdir(ppath)
            for phase in phases:
                cpath = os.path.join(ppath, phase)
                # dpath = os.path.join(cpath, 'data')
                # lpath = os.path.join(dpath, 'label')
                assert os.path.exists(cpath), ' label dir not exist under: ' + cpath

                sids = os.listdir(cpath)
                for sid in sids:
                    sid_path = os.path.join(cpath, sid)
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
        print sid_label_list
        return sid_label_list

    def _read(self):
        sid_path = self.pids[self.cursor]
        self.sid = sid_path
        data, label, raw, voxel_vol = self._read_sid(sid_path)
        return data, label, raw, voxel_vol

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
        img_nrrd = [item for item in items if item.endswith('.nrrd') and not item.endswith('label.nrrd')]
        lbl_nrrd = [item for item in items if item.endswith('label.nrrd')]
        assert len(img_nrrd) == 1 and len(lbl_nrrd) == 1, 'img or lbl nrrd not the only one'
        img_nrrd = img_nrrd[0]
        lbl_nrrd = lbl_nrrd[0]
        try:
            t = nrrd.read(os.path.join(sid_path, img_nrrd))
            t_lbl = nrrd.read(os.path.join(sid_path, lbl_nrrd))
            imgs = t[0]  # 由于对数据有去骨和不去骨的预处理，为了便于后面显示画图，所以有imgs(有可能经过预处理)，raws（未经过处理的原始数据）
            raws = np.transpose(t[0], (2, 0, 1))  # N*512*512
            lbls = t_lbl[0]
            
	    # calculate physical volume using space directions info stored in .nrrd
	    space_matrix = np.zeros((3, 3))
	    space_matrix[0] = np.asarray(t_lbl[1]['space directions'][0]).astype('float32')
	    space_matrix[1] = np.asarray(t_lbl[1]['space directions'][1]).astype('float32')
	    space_matrix[2] = np.asarray(t_lbl[1]['space directions'][2]).astype('float32')
	    # calculate voxel volume as the determinant of spacing matrix
	    voxel_vol = np.linalg.det(space_matrix)
	    print voxel_vol
            del t
            del t_lbl
        except Exception as e:
            print e
            return None, None, None, None

        # lbls = self.label_convert_brainRegion(lbls)  # 512 512 N
        assert imgs.shape[-1] > 2
        assert lbls.shape[-1] > 2

        # resize the images and labels to designated size (defined in config)
        if np.shape(imgs)[0:2] != (self.img_shape[0], self.img_shape[1]):
            fx = float(self.img_shape[0]) / float(np.shape(imgs)[0])
            fy = float(self.img_shape[1]) / float(np.shape(imgs)[1])
            new_imgs = np.zeros((self.img_shape[0], self.img_shape[1], np.shape(lbls)[2]))
            for batch_num in range(np.shape(lbls)[2]):
                 new_imgs[:, :, batch_num] = cv2.resize(imgs.astype('float32')[:, :, batch_num], None, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            imgs = new_imgs.astype('int16')
	    raws = imgs.transpose(2, 0, 1) # N*512*512
        if np.shape(lbls)[0:2] != (self.img_shape[0], self.img_shape[1]):
            fx = float(self.img_shape[0]) / float(np.shape(lbls)[0])
            fy = float(self.img_shape[1]) / float(np.shape(lbls)[1])
            new_lbls = np.zeros((self.img_shape[0], self.img_shape[1], np.shape(lbls)[2]))
            for batch_num in range(np.shape(lbls)[2]):
                new_lbls[:, :, batch_num] = cv2.resize(lbls.astype('int16')[:, :, batch_num], None, None, fx=fx, fy=fy,
                                                       interpolation=cv2.INTER_NEAREST)
            lbls = new_lbls
	
        #imgs = get_off_bone(imgs, thresh=150)
        # imgs = get_off_negative(imgs, thresh=-200)

        if self.mode == 'tile':
            imgs = imgs.reshape((self.img_shape[0], self.img_shape[1], 1, -1))
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

        # Positive and negative sample balance
        if self.is_train:
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
        if self.if_binary:
            lbls[lbls > 0] = 1.
            masked_img = imgs[:, 0, :, :] * lbls
            lbls[masked_img < 130] = 0.
        imgs = window_convert(imgs, self.window_center, self.window_width)
        # print np.histogram(lbls)
        return imgs, lbls, raws, voxel_vol


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
            self.data, self.label, raw, self.voxel_vol = self._read()
            #print '%%%%%%%%%%%%%%%%', type(self.data),np.shape(self.data)
            return {self.data_name: self.data,
                    self.label_name: self.label,
                    'pid': self.pids[self.cursor].split('/')[-1],
                    'raw': raw,
		            'voxel_vol': self.voxel_vol
                    }
        else:
            return None
