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