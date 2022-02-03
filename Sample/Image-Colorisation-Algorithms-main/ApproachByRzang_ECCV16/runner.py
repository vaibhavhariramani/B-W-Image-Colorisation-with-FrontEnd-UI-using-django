import cv2
import sys
import os

from eccv16 import ECCVGenerator
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import eccv16
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import argparse
import matplotlib.pyplot as plt

from execute import postprocess_tens, preprocess_img
image_path = sys.argv[1]
image_name = sys.argv[2]

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('colorization.pth',map_location='cpu',check_hash=True))
	return model
# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()

#load image
img = cv2.imread(str(image_path))

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())


image_save_path = image_path.replace(image_name, "temp.png")
# cv2.imwrite(str(image_save_path), out_img_siggraph17)
plt.imsave(str(image_save_path), out_img_eccv16)
print('media/temp.png')