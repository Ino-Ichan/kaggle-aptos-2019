import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import cv2

from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt

import gc
from tqdm import tqdm

import os
import scipy as sp
from functools import partial

# basic
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPool2D, Softmax
from keras import optimizers
from keras.utils import Sequence

# augmentation
import imgaug.augmenters as iaa
import imgaug as ia


"""
crop_image_from_gray method is quated from this excellent kernel (https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping)!!!
"""


class PreTrainDataGenerator(Sequence):
    """
    For pretraining 2015-train data.
    """
    def __init__(self, df, batch_size, img_size=224, data_path="../data/resized_train_15/", augmentation=True, shuffle=True):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentation = augmentation
        
        self.data_path = data_path
        
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.df))
        
        if self.augmentation:
            self.aug_seq = self.get_imgaug_seq()
        
    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        img_paths = self.df.iloc[inds]["image"].values
        
        # ここに画像を格納する
        images_list = []
        # ラベルのリスト
        labels = self.df.iloc[inds]["level"].values
        sigmaX = 10
        for path in img_paths:
            img = cv2.imread(os.path.join(self.data_path, path+".jpg"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.crop_image_from_gray(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0) , sigmaX) ,-4 ,128)
            # 一旦、listにしておく
            images_list.append(list(img/255))
        images_list = np.array(images_list)
        
        labels_list = labels
        
        if self.augmentation:
            images_list = self.aug_seq.augment_images(images_list)
        
        return images_list, labels_list
    
    def __len__(self):
        return math.ceil(len(self.df)/self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def crop_image_from_gray(self, img, tol=7):
        """
        Applies masks to the orignal image and 
        returns the a preprocessed image with 
        3 channels
        """
        # If for some reason we only have two channels
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        # If we have a normal RGB images
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
    
    def get_imgaug_seq(self):
        ia.seed(1)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90([0, 1, 2, 3]),
        ], random_order=True)
        return seq
    
    
class TrainDataGenerator(Sequence):
    """
    For training 2019-train data
    """
    def __init__(self, df, batch_size, img_size=224, data_path="../data/train_images/train_images/", augmentation=False, shuffle=False):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmentation = augmentation
        
        self.data_path = data_path
        
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.df))
        
        if self.augmentation:
            self.aug_seq = self.get_imgaug_seq()
        
    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        img_paths = self.df.iloc[inds]["id_code"].values
        
        # ここに画像を格納する
        images_list = []
        # ラベルのリスト
        labels = self.df.iloc[inds]["diagnosis"].values
        sigmaX = 10
        for path in img_paths:
            img = cv2.imread(os.path.join(self.data_path, path+".png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.crop_image_from_gray(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0) , sigmaX) ,-4 ,128)
            # 一旦、listにしておく
            images_list.append(list(img/255))
        images_list = np.array(images_list)
        
        labels_list = labels
        
        if self.augmentation:
            images_list = self.aug_seq.augment_images(images_list)
        
        return images_list, labels_list
    
    def __len__(self):
        return math.ceil(len(self.df)/self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def crop_image_from_gray(self, img, tol=7):
        """
        Applies masks to the orignal image and 
        returns the a preprocessed image with 
        3 channels
        """
        # If for some reason we only have two channels
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        # If we have a normal RGB images
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
    
    def get_imgaug_seq(self):
        ia.seed(1)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90([0, 1, 2, 3]),
        ], random_order=True)
        return seq
    
    
class TestDataGenerator(Sequence):
    """
    For validation or test of 2019-data
    """
    def __init__(self, df, batch_size, img_size=224, data_path="../data/train_images/train_images/"):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.data_path = data_path
        
        
        self.indices = np.arange(len(self.df))
        
    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        img_paths = self.df.iloc[inds]["id_code"].values
        
        # ここに画像を格納する
        images_list = []
        
        sigmaX = 10
        for path in img_paths:
            img = cv2.imread(os.path.join(self.data_path, path+".png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.crop_image_from_gray(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0) , sigmaX) ,-4 ,128)
            # 一旦、listにしておく
            images_list.append(list(img/255))
        images_list = np.array(images_list)
        

        
        return images_list
    
    def __len__(self):
        return math.ceil(len(self.df)/self.batch_size)
            
    def crop_image_from_gray(self, img, tol=7):
        """
        Applies masks to the orignal image and 
        returns the a preprocessed image with 
        3 channels
        """
        # If for some reason we only have two channels
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        # If we have a normal RGB images
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img