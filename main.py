import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import *

prefix = "med"

if prefix == "low":
    PATH = 'images/Low concentration 1.tiff'
elif prefix == "med":
    PATH = 'images/Medium concentration 1.tiff'
elif prefix == "high":
    PATH = 'images/High concentration 1.tiff'

# Leer tiff y convertirlo en np.array
im = read_tiff(PATH)
im = im[:60] if prefix=="med" else im
im = isotropic_interpolation(im)
segmentate_matrix(im, prefix)
