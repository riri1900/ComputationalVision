import numpy as np
import cv2
from scipy.ndimage import convolve

def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    #initializing
    dispM = np.zeros_like(im1, dtype=float)
    distance = np.zeros_like(im1,dtype=float)
    disps = np.zeros((im1.shape[0], im1.shape[1], maxDisp+1), dtype=np.float32)
    #creating a square mask using winowSize
    mask = np.ones((windowSize,windowSize))
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    for disp in range (maxDisp+1):
        #move to the right by disp
        im2_d = np.roll(im2,disp,axis=1)
        #convolution
        disps[:,:,disp] = convolve((im1 - im2_d)**2, mask, mode='constant' )


    dispM = np.argmin(disps, axis=2)

    return dispM
    
