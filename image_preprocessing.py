#Importing required libraries
from __future__ import division

import numpy as np
import pandas as pd
import os
from glob import glob
import re

from tqdm import tqdm

from skimage import io
from skimage.color import gray2rgb
from skimage.util import crop
from skimage.transform import rescale, resize

import warnings
warnings.filterwarnings('ignore')

#building required functions

def crop_image(img):
    """Remove white borders from image"""
    croped_img = img.copy()
    # 1 - remove left white border    
    j = 0
    while np.prod(croped_img[:,j] == 255):
        j+=1
    croped_img = croped_img[:,j:,:]
    
    # 2 - remove top white border   
    i = 0
    while np.prod(croped_img[i,:] == 255):
        i+=1
    croped_img = croped_img[i:,:,:]
    
    # 3 - remove right white border    
    j = croped_img.shape[1]-1
    while np.prod(croped_img[:,j] == 255):
        j-=1
    croped_img = croped_img[:,:j,:]
    
    # 4 - remove top white border   
    i = croped_img.shape[0]-1
    while np.prod(croped_img[i,:] == 255):
        i-=1
    croped_img = croped_img[:i,:,:]
    
    return croped_img

def rescaleByWidth(img, width=299):
    """Get image with shortest side = 'width'"""
    scale_rate = width / min([img.shape[0],img.shape[1]])
    return rescale(img.copy(), scale_rate)

def frame_rect(img):
    """Get rectangle frame"""
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    long_side = max(img_height, img_width)
    short_side = min(img_height, img_width)

    cut_border = int((long_side - short_side) / 2.)
    
    crop_param = []
    
    if img_height >= img_width:
        crop_param = ((cut_border, cut_border), (0,0), (0,0))
    else:
        crop_param = ((0,0), (cut_border, cut_border), (0,0))
        
    return crop(img.copy(), crop_param)

def frame_rect_part(img, part_rate = 0.5):
    """Get part of rectangle image"""
    cropped_img = frame_rect(img)
    
    img_height = cropped_img.shape[0]
    img_width = cropped_img.shape[1]
    
    long_side = max([img_height, img_width])
    
    border = int(long_side/2.) - int(long_side/2.*part_rate)
    
    crop_params = ((border, border), (border, border), (0, 0))
    
    return crop(cropped_img.copy(), crop_params)

def process_img(img, toCrop=True, toFraneRectPart=True, part_rate=1., toScale=True, scale_width=299):
    """Process image applying different functions"""
    img_processed = img.copy()    
    if toCrop:
        img_processed = crop_image(img_processed)        
    if toFraneRectPart:
        img_processed = frame_rect_part(img_processed, part_rate)
    if toScale:
        img_processed = rescaleByWidth(img_processed, scale_width)                
    return img_processed

#Processing train images

# get names of train images
image_names = glob('../Data/Train_vf/*/*.*')  **

# folder to save processed images
train_processed_folder = 'Train_vf_processed_08'

# create folder to save processed images if it doesn't exists
if not os.path.exists(os.path.join('../Data/', train_processed_folder)):  ** 
    os.mkdir(os.path.join('../Data/', train_processed_folder))
    
    # process every image in train folder and save processed image in new folder
    for image_name in tqdm(image_names):
        try:
            # create folder for classif it doen't exists
            class_name = re.split(r'/', image_name)[-2]
            if not os.path.exists(os.path.join('../Data/', train_processed_folder,class_name)):  **
                os.mkdir(os.path.join('../Data/', train_processed_folder,class_name))

            # read image
            img = io.imread(image_name)

            # create name for new image - it's needed to save processed image
            image_name = re.sub(r'Train_vf', train_processed_folder, image_name)  

            # if image is grayslace then make it to be RGB
            if len(img.shape) < 3:
                img = gray2rgb(img)

            # process image
            img = process_img(img, part_rate=0.8, scale_width=299)

            # save new image
            io.imsave(image_name, img)
        except Exception:
            continue
else:
    print("Folder is already exists. Remove this folder or set another name")

#Preparing Validation Set

#Load file with validation labels
valid = pd.read_excel('../Data/Pattern_Solution_file - Public_Test.xlsx', index_col='IMG_ID')  **
valid.head()

#Create folder with preprocessed validation images
valid_folder_origin = '../Data/Test_vf/Public_Test/'

# name for folder to save processed validation set
valid_folder_name = 'Valid_vf_processed_08' 

# create validation folder if it doesn't exists
valid_folder_prepared = os.path.join('../Data/', valid_folder_name)  **
if not os.path.exists(valid_folder_prepared):
    os.mkdir(valid_folder_prepared)

#process all images
for valid_img_name in tqdm(valid.index):
    # read image
    valid_img = io.imread(os.path.join(valid_folder_origin, valid_img_name))
    
    # convert to RGB if it is grayscale
    if len(valid_img.shape) < 3:
        valid_img = gray2rgb(valid_img)
    
    # process image
    valid_img = process_img(valid_img, part_rate=0.8, scale_width=299)
    
    # save image
    valid_label = valid.get_value(valid_img_name, 'Actual Pattern')
    
    if not os.path.exists(os.path.join(valid_folder_prepared, valid_label)):
        os.mkdir(os.path.join(valid_folder_prepared, valid_label))
    
    io.imsave(os.path.join(valid_folder_prepared, valid_label, valid_img_name), valid_img)

#Preparing Test Set

#load files with test labels
test = pd.read_excel('../Data/Pattern_Submission_file.xlsx', index_col='IMG_ID')  **
test.head()

#Create folder with preprocessed test images
test_folder_origin = '../Data/Test_vf/Private_Test/'
test_folder_name = 'Test_vf_processed_08'

test_folder_prepared = os.path.join('../Data/', test_folder_name)
if not os.path.exists(test_folder_prepared):
    os.mkdir(test_folder_prepared)
        
# we create folder in folder because it's needed for data generator in Keras
test_folder_prepared = os.path.join('../Data/', test_folder_name, test_folder_name)
if not os.path.exists(test_folder_prepared):
    os.mkdir(test_folder_prepared)


#process images
for test_img_name in tqdm(test.index):
    # read image
    test_img = io.imread(os.path.join(test_folder_origin, test_img_name))
    
    # convert to RGB if it is grayscale
    if len(test_img.shape) < 3:
        test_img = gray2rgb(test_img)
        
    # process image       
    test_img = process_img(test_img, part_rate=0.8, scale_width=299)
    
    # save image
    io.imsave(os.path.join(test_folder_prepared, test_img_name), test_img)