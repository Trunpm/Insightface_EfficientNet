from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm
import os



def load_mx_rec(rec_path,savefold):
    save_path = savefold+'/imgs'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path+'/'+'train.idx'), str(rec_path+'/'+'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path+'/'+str(label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        img.save(label_path+'/'+'{}.jpg'.format(idx), quality=100)

def load_bin(path, savepath, image_size=[112,112]):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(savepath+'/'+'{}.jpg'.format(i), quality=100)
    np.save(savepath.split('/')[-1]+'_list', np.array(issame_list))

if __name__ == '__main__':
    mainfold = './faces_emore'
    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    savefold = './faces_emore_imgs'
    load_mx_rec(mainfold,savefold)
    for i in range(len(bin_files)):
        load_bin(mainfold+'/'+bin_files[i]+'.bin', savepath = savefold+'/'+bin_files[i], image_size=[112,112])
