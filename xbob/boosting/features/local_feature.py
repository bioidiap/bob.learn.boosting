import numpy as np
import math
#import baseloss
import random
#import weaklearner
from pylab import *

class lbp_feature():

    def __init__(self, ftype):
        self.ftype = ftype


    def integral_img(self,img):
        int1 = cumsum(img,0)
        int2 = cumsum(int1,1)
        return int2

    def get_features(self,img,cx,cy):
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
                blk_int = int_img[sy+1:,sx+1:] + int_img[0:-(sy+1),0:-(sx+1)] - int_img[sy+1:,0:-(sx+1)] - int_img[0:-(sy+1),sx+1:]
                blk_int = int_img[sy:,sx:] + int_img[0:-sy,0:-sx] - int_img[sy:,0:-sx] - int_img[0:-sy,sx:]
                fmap_dimy, fmap_dimx = blk_int.shape -2
                coord = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]

                if(self.ftype == 'lbp'):
                    fmap = self.lbp(coord, fmap_dimx, fmap_dimy, blk_int)
                else(self.ftype == 'tlbp'):
                    fmap = self.tlbp(coord, fmap_dimx, fmap_dimy, blk_int)
                else(self.ftype == 'dlbp'):
                    fmap = self.dlbp(coord, fmap_dimx, fmap_dimy, blk_int)
                else(self.ftype == 'mlbp'):
                    fmap = self.mlbp(coord, fmap_dimx, fmap_dimy, blk_int)

                vec = np.reshape(fmap,fmap.shape[0]*fmap.shape[1],1)
                fvec = np.hstack((fvec,vec))
        return fvec

    def lbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        num_neighbours = 8
        blk_center = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
        for ind in range(num_neighbours):
            fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= blk_center)
        return fmap



    def tlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        num_neighbour = 8

        for ind in range(num_neighbours):
            comp_img = blk_int[coord[(ind+1)%num_neighbour][0]:coord[(ind+1)%num_neighbour][0] + fmap_dimy,coord[(ind+1)%num_neighbour][1]:coord[(ind+1)%num_neighbour][1] + fmap_dimx]
            fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= comp_img)
        return fmap



    def dlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        pc = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
        num_neighbours = 8
        fmap = np.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours/2):
            pi = blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
            pi4 = blk_int[coord[ind+4][0]:coord[ind+4][0]+ fmap_dimy,coord[ind+4][1]:coord[ind+4][1] + fmap_dimx]
            fmap = fmap + (2**ind)*((pi-pc)*(pi4 - pc) > 0) + (4**ind)*(abs(pi - pc) >= abs(pi4 -pc))
        return fmap



    def mlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        num_neighbours = 8
        pm = np.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours):
            pm = pm + blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
        pm = pm/num_neighbours

        fmap = np.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours):
            pi = blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
            fmap = fmap + (2**ind)*(pi >= pm)
        return fmap
            
       
    def get_feature_number(self, dimy, dimx, scale_y, scale_x)
        img = np.zeros([dimy, dimx])
        feature_vector = self.get_features(img, scale_y, scale_x)
        return feature_vector.shape


