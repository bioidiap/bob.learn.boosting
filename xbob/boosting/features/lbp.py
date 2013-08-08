import numpy as np
import math
#import baseloss
import random
#import weaklearner
from pylab import *

class mlbp():

    def __init__(self, ftype):
        self.ftype = ftype


    def test_func():
        a = np.ones([20,24])
        b = compute_mlbp(a,6,7)



    def integral_img(self,img):
        int1 = cumsum(img,0)
        int2 = cumsum(int1,1)
        return int2

    def compute_mlbp(self,img,cx,cy):
        # Initializations
        int_imgc = self.integral_img(img)
        rows, cols = img.shape
        int_img = np.zeros([rows+1,cols+1])
        int_img[1:,1:] = int_imgc 
        num_neighbours = 8
        fvec = np.empty(0, dtype = 'uint8')

        for isx in range(cx):
            for isy in range(cy):
                sx = isx +1
                sy = isy +1
                blk_int = int_img[sy:,sx:] + int_img[0:-sy,0:-sx] - int_img[sy:,0:-sx] - int_img[0:-sy,sx:]
                blk_dimy, blk_dimx = blk_int.shape
                fmap_dimx = blk_dimx - 2
                fmap_dimy = blk_dimy - 2
                fmap = np.zeros([fmap_dimy,fmap_dimx])
                coord = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]
                blk_center = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
                for ind in range(num_neighbours):
                    if(self.ftype == 'lbp'):
                        fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= blk_center)
                    elif(self.ftype == 'tlbp'):
                        comp_img = blk_int[coord[(ind+1)%8][0]:coord[(ind+1)%8][0] + fmap_dimy,coord[(ind+1)%8][1]:coord[(ind+1)%8][1] + fmap_dimx]
                        fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= comp_img)
                vec = np.reshape(fmap,fmap.shape[0]*fmap.shape[1],1)
                fvec = np.hstack((fvec,vec))
        return fvec




