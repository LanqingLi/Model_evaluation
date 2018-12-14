#!/home/lanruo_wan/venv python
#_*_coding:utf-8_*_
import numpy as np
import sys, os
from mxnet.io import DataIter
import numpy.random as npr
import matplotlib.pyplot as plt
from window_converter import window_convert, window_convert_light
import nrrd
import cv2
from config import config
#from cv_toolkit.resample.crop import CropTool, resample_crop_img
# from preprocessing.off_bone import get_off_bone

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
                 if_binary=True,
                 #if_crop=False
                 ):
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
        self.seqlen = config.seqlen
        #self.if_crop = if_crop
        self.data, self.label, _, self.voxel_vol, self.pixel_area = self._read()

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
        data, label, raw, voxel_vol, pixel_area = self._read_sid(sid_path)
        return data, label, raw, voxel_vol, pixel_area

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
            # raws = np.transpose(t[0], (2, 0, 1))  # N*512*512
            raws = imgs.transpose(2, 0, 1)  # N*512*512
            lbls = t_lbl[0]

            # calculate physical volume and area using space directions info stored in .nrrd
            space_matrix = np.zeros((3, 3))
            slice_matrix = np.zeros((2, 2))
            space_matrix[0] = np.asarray(t_lbl[1]['space directions'][0]).astype('float32')
            space_matrix[1] = np.asarray(t_lbl[1]['space directions'][1]).astype('float32')
            space_matrix[2] = np.asarray(t_lbl[1]['space directions'][2]).astype('float32')
            slice_matrix[0] = np.asarray(t_lbl[1]['space directions'][0][:2]).astype('float32')
            slice_matrix[1] = np.asarray(t_lbl[1]['space directions'][1][:2]).astype('float32')
            # calculate voxel volume as the determinant of the space matrix, pixel area as the det of the slice matrix
            pixel_area = np.linalg.det(slice_matrix)
            voxel_vol = np.linalg.det(space_matrix)
            print 'pixel_area: {}'.format(pixel_area)
            print 'voxel_vol: {}'.format(voxel_vol)
            del t
            del t_lbl
        except Exception as e:
            print e
            return None, None, None, None, None

        # lbls = self.label_convert_brainRegion(lbls)  # 512 512 N
        assert imgs.shape[-1] > 2
        assert lbls.shape[-1] > 2

        # resize the images and labels to designated size (defined in config)
        if np.shape(imgs)[0:2] != (self.img_shape[0], self.img_shape[1]):
            fx = float(self.img_shape[0]) / float(np.shape(imgs)[0])
            fy = float(self.img_shape[1]) / float(np.shape(imgs)[1])
            new_imgs = np.zeros((self.img_shape[0], self.img_shape[1], np.shape(lbls)[2]))
            for batch_num in range(np.shape(lbls)[2]):
                new_imgs[:, :, batch_num] = cv2.resize(imgs.astype('float32')[:, :, batch_num], None, None, fx=fx,
                                                       fy=fy, interpolation=cv2.INTER_LINEAR)
            imgs = new_imgs.astype('int16')
            raws = imgs.transpose(2, 0, 1)  # N*512*512
            voxel_vol = voxel_vol / (fx * fy)
            pixel_area = pixel_area / (fx * fy)
        if np.shape(lbls)[0:2] != (self.img_shape[0], self.img_shape[1]):
            fx = float(self.img_shape[0]) / float(np.shape(lbls)[0])
            fy = float(self.img_shape[1]) / float(np.shape(lbls)[1])
            new_lbls = np.zeros((self.img_shape[0], self.img_shape[1], np.shape(lbls)[2]))
            for batch_num in range(np.shape(lbls)[2]):
                new_lbls[:, :, batch_num] = cv2.resize(lbls.astype('int16')[:, :, batch_num], None, None, fx=fx, fy=fy,
                                                       interpolation=cv2.INTER_NEAREST)
            lbls = new_lbls

        # imgs = get_off_bone(imgs, thresh=150)
        # imgs = get_off_negative(imgs, thresh=-200)

        if self.mode == 'tile':
            imgs = imgs.reshape((self.img_shape[0], self.img_shape[1], 1, -1))
            imgs = np.tile(imgs, (1, 1, 1, 1))  # 512,512,1,N
            imgs = np.transpose(imgs, (3, 2, 0, 1))  # N,3,512,512
        else:
            image = [imgs[:, :, i:i + 3] for i in range(imgs.shape[-1] - 2)]
            imgs = np.array(image).transpose((0, 3, 1, 2))
            lbls = lbls[:, :, 1:lbls.shape[-1] - 1]

        #reshaped_mean = self.mean.reshape(1, 3, 1, 1)
        #imgs = imgs - reshaped_mean  # imgs (N, 3, 512, 512)
        raws = np.repeat(raws[:, np.newaxis, :, :], axis=1, repeats=3)  # N*1*512*512

        lbls = np.transpose(lbls, (2, 0, 1))  # N 512 512

        # Positive and negative sample balance
        if self.is_train:
            # if self.seqlen == 1:
            #     lblss = []
            #     imgss = []
            #     rawss = []
            #     keep = []
            #     pos_count = 0
            #     tot_count = 0
            #     for lbl, img, raw in zip(lbls, imgs, raws):
            #         if np.max(lbl) > 0:
            #             lblss.append(lbl)
            #             imgss.append(img)
            #             rawss.append(raw)
            #             pos_count += 1
            #         else:
            #             keep.append(tot_count)
            #         tot_count += 1
            #     if lbls.shape[0] - (self.negative_mining_ratio + 1) * pos_count <= 0:  # 若一个sid中，正样本的数量多余负样本数量，则不作样本平衡
            #         lblss = lbls
            #         imgss = imgs
            #         rawss = raws
            #     else:
            #         bg_inds = npr.choice(keep, size=self.negative_mining_ratio * pos_count,
            #                              replace=False)  # 若正样本数量少于负样本，则随机选择出和正样本同样数量的负样本
            #         lbl = lbls[bg_inds]
            #         img = imgs[bg_inds]
            #         raw = raws[bg_inds]
            #         for label, image, raww in zip(lbl, img, raw):
            #             lblss.append(label)
            #             imgss.append(image)
            #             rawss.append(raww)
            #     lblss = np.array(lblss)
            #     imgss = np.array(imgss)
            #     rawss = np.array(rawss)
            #     # 若正样本数为0且negative_mining_ratio=1,前面的样本平衡中增加的负样本也为0, 也就是说该sid是正常人，没有出血，
            #     # 为了保持数据迭代器输出的一致性，选取第一张负样本图像，后面的if将其复制成batch_size份用于输出
            #     if len(lblss) == 0 and len(imgss) == 0:
            #         print sid_path
            #         lblss = lbls[0, :, :]
            #         imgss = imgs[0, :, :, :]
            #         rawss = raws[0, :, :, :]
            #
            #         lblss = lblss[np.newaxis, :, :]
            #         imgss = imgss[np.newaxis, :, :, :]
            #         rawss = rawss[np.newaxis, :, :, :]
            #     if lblss.shape[0] == 1 and imgss.shape[0] == 1:
            #         lblss = np.tile(lblss, (self.batch_size, 1, 1))
            #         imgss = np.tile(imgss, (self.batch_size, 1, 1, 1))
            #         rawss = np.tile(rawss, (self.batch_size, 1, 1, 1))
            #
            #     index = [i for i in range(lblss.shape[0])]
            #     npr.shuffle(index)
            #     lbls = lblss[index]
            #     imgs = imgss[index]
            #     raws = rawss[index]

            #else:
            assert self.seqlen > 0 and self.seqlen % 2 == 1, 'sequence length must be an odd postive integer!'
            pos_lblss = []
            pos_imgss = []
            pos_rawss = []
            neg_lblss = []
            neg_imgss = []
            neg_rawss = []
            pos_count = 0
            neg_list = []
            neg_count = 0
            slice_num = lbls.shape[0]
            assert slice_num >= self.seqlen, 'the number of slices of a series must be greater than or equal to %d' % (
            self.seqlen)
            for tot_count, (lbl, img, raw) in enumerate(zip(lbls, imgs, raws)):
                if tot_count < self.seqlen / 2 or tot_count > slice_num - self.seqlen / 2 - 1:
                    if self.exclude_boundary:
                        # exclude boundary cases
                        continue
                    else:
                        # if we include boundary cases, pad all out-of-boundary slices as zero
                        padded_img = np.zeros(shape=(self.seqlen, img.shape[0], img.shape[1]))
                        if tot_count < self.seqlen / 2:
                            padded_img[self.seqlen / 2 - tot_count: self.seqlen] = imgs[
                                                                                   : tot_count + self.seqlen / 2 + 1]
                            pos_lblss.append(lbl)
                            pos_imgss.append(padded_img)
                            pos_rawss.append(raw)
                            pos_count += 1
                        else:
                            padded_img[: self.seqlen * 3 / 2 - tot_count] = imgs[tot_count - self.seqlen / 2:]
                            pos_lblss.append(lbl)
                            pos_imgss.append(padded_img)
                            pos_rawss.append(raw)
                            pos_count += 1

                else:
                    if np.max(lbl) > 0:
                        pos_lblss.append(lbl)
                        pos_imgss.append(imgs[tot_count - self.seqlen / 2: tot_count + self.seqlen / 2 + 1])
                        pos_rawss.append(raw)
                        pos_count += 1
                    else:
                        neg_lblss.append(lbl)
                        neg_imgss.append(imgs[tot_count - self.seqlen / 2: tot_count + self.seqlen / 2 + 1])
                        neg_rawss.append(raw)
                        neg_list.append(neg_count)
                        neg_count += 1
            if lbls.shape[0] - (self.negative_mining_ratio + 1) * pos_count <= 0:  # 若一个sid中，正样本的数量多余负样本数量，则不作样本平衡
                lblss = pos_lblss + neg_lblss
                imgss = pos_imgss + neg_imgss
                rawss = pos_rawss + neg_rawss
            else:
                bg_inds = npr.choice(neg_list, size=self.negative_mining_ratio * pos_count,
                                     replace=False)  # 若正样本数量少于负样本，则随机选择出和正样本同样数量的负样本
                #print bg_inds
                lblss = pos_lblss + list(np.array(neg_lblss)[bg_inds])
                imgss = pos_imgss + list(np.array(neg_imgss)[bg_inds])
                rawss = pos_rawss + list(np.array(neg_rawss)[bg_inds])

            lblss = np.array(lblss)
            imgss = np.array(imgss)
            rawss = np.array(rawss)
            #print lblss.shape, imgss.shape, rawss.shape

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

            imgs = imgs.reshape(imgs.shape[0], -1, imgs.shape[3], imgs.shape[4])

        else:
            new_imgs = np.repeat(imgs[:, :, :, :], axis=1, repeats=self.seqlen)  # N * self.seqlen * 512 * 512
            for chn_num in range(self.seqlen):
                batch_size = new_imgs.shape[0]
                print chn_num
                if chn_num < self.seqlen / 2:
                    # pad all out-of-boundary slices to zero by default
                    new_imgs[:(self.seqlen / 2 - chn_num), chn_num:chn_num+1, :, :] = 0
                    new_imgs[(self.seqlen / 2 - chn_num):, chn_num:chn_num+1, :, :] = imgs[:(batch_size - self.seqlen / 2 + chn_num)]
                elif chn_num > self.seqlen / 2:
                    new_imgs[(self.seqlen * 3 / 2 - chn_num):, chn_num:chn_num+1, :, :] = 0
                    new_imgs[:(self.seqlen / 2 + batch_size - chn_num), chn_num:chn_num+1, :, :] = imgs[(chn_num - self.seqlen / 2):]
                else:
                    new_imgs[:, chn_num:chn_num+1, :, :] = imgs
            imgs = new_imgs
            del new_imgs

        print lbls.shape, imgs.shape, raws.shape
        # add randomly cropped data
        # if self.if_crop:
        #     crop_lbls = lbls.copy()
        #     crop_imgs = imgs.copy()
        #     #crop_raws = raws.copy()
        #     count = 0
        #     for lbl, img in zip(lbls, imgs):
        #             crop_tool = CropTool(img_array=img, tgt_shape=config.tgt_crop_shape, min_shape=config.min_crop_shape,
        #                                  crop_dof=config.crop_dof, fix_crop_shape=config.fix_crop_shape)
        #             new_img, new_label = crop_tool.get_crop_img_label_uniform(lbl)
        #             slice_count = 0
        #             new_resample_label = resample_crop_img(new_label.astype('int16'), tgt_shape=(self.img_shape[0], self.img_shape[1]), interpolator=cv2.INTER_NEAREST)
        #             for new_img_slice in new_img:
        #                 new_img_resample_slice = resample_crop_img(new_img_slice.astype('int16'), tgt_shape=(self.img_shape[0], self.img_shape[1]))
        #
        #                 crop_imgs[count][slice_count] = new_img_resample_slice
        #                 slice_count += 1
        #             crop_lbls[count] = new_resample_label
        #             count += 1
        #
        #     lbls = np.concatenate((lbls, crop_lbls), axis=0)
        #     imgs = np.concatenate((imgs, crop_imgs), axis=0)
        # if lbls.shape[0] == 1 and imgs.shape[0] == 1:  #防止图像数目小于batch_size
        #     lbls = np.tile(lbls, (self.batch_size, 1, 1)) #比如 a = np.array([0,1,2]),    np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍，即最终得到
        #     imgs = np.tile(imgs, (self.batch_size, 1, 1, 1))
        #     raws = np.tile(raws, (self.batch_size, 1, 1, 1))

        # print 'imgs.shape in _read_sid func: ', self.sid,imgs.shape[0], imgs.shape
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
            masked_img = imgs[:, self.seqlen/2, :, :] * lbls
            lbls[masked_img < 130] = 0.
        #imgs = window_convert(imgs, self.window_center, self.window_width)
        imgs = window_convert_light(imgs, self.window_center, self.window_width)
        print 'img_shape = ', imgs.shape
        print 'label_shape= ', lbls.shape
        return imgs, lbls, raws, voxel_vol, pixel_area

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
            self.data, self.label, raw, self.voxel_vol, self.pixel_area = self._read()
            # print '%%%%%%%%%%%%%%%%', type(self.data),np.shape(self.data)
            return {self.data_name: self.data,
                    self.label_name: self.label,
                    'pid': self.pids[self.cursor].split('/')[-1],
                    'raw': raw,
                    'voxel_vol': self.voxel_vol,
                    'pixel_area': self.pixel_area
                    }
        else:
            return None
