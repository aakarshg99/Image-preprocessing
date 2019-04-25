import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from skimage.util import crop

img = "./test.jpg"

#plt.imshow(img)
#plt.show()


def crop_image(img_path):
	img    = cv2.imread(img_path)
	img_height  = img.shape[0]
	img_width   = img.shape[1]
	cut_border  = int(img_height/2.)
	crop_params = ((0,cut_border), (0,0), (0,0))
	return crop(img.copy(), crop_params)

#image_folder = glob('../Data/Train_vf/*/*.*')

"""img = crop_image(img)
plt.imshow(img)
plt.show()"""
source_folder_path = "./Train_old_collar_type/vneck"
destination_folder = "./Train_old_collar_type_cropped/vneck"
list_img           = os.listdir(source_folder_path)

for single_img in list_img:
    source_img_path   = source_folder_path+"/"+single_img
    destint_img_path  = destination_folder+"/"+single_img+"_cropped.jpg"
    cropped_image     = crop_image(source_img_path)
    #cropped_image     = cv2.resize(cropped_image, (350, 350))
    cv2.imwrite(destint_img_path, cropped_image)