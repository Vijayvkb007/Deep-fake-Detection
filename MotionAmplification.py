import os
import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

# Set hyperparameters

# Video magnification factor
ALPHA = 50.0

# Gaussian pyramid level of which to apply magnification 
LEVEL = 4

# Temporal filter parameters
f_lo = 50/60
f_hi = 60/60

# optional : override fs
MANUAL_FS = None
VIDEO_FS = None

# Video frame scale factor
SCALE_FACTOR = 1.0

# color space function
def rgb2yiq(rgb):
    """Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        Inputs:
            rgb - (N, M, 3) rgb image
        Returns:
            yiq - (N, M, 3) yiq image"""