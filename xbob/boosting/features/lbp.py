import numpy as np
import math
#import baseloss
import random
#import weaklearner
from pylab import *


def compute_blkimgs(img,Cx,Cy):
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    blk_imgs = []
    for cx in range(Cx):
	for cy in range(Cy):
            print 'cx %d cy %d' % (cx,cy) 
            num_rows = img_rows - (cy)
            num_cols = img_cols - (cx)
	    curr_img = np.zeros([num_rows,num_cols])
	    for i in range(cx+1):
		for j in range(cy+1):
                    print i
                    print j
		    curr_img  = curr_img + img[i:i+num_rows,j:j+num_cols]
	    blk_imgs.append(curr_img)
    return blk_imgs
   
def test_func():
    a = np.ones([20,24])
    b = compute_mlbp(a,6,7)



def integral_img(img):
    int1 = cumsum(img,0)
    int2 = cumsum(int1,1)
    return int2

def compute_mlbp(img,cx,cy):
    int_imgc = integral_img(img)
    rows = img.shape[0]
    cols = img.shape[1]
    int_img = np.zeros([rows+1,cols+1])
    int_img[1:,1:] = int_imgc 
    num_neighbours = 8
    for isx in range(cx):
	for isy in range(cy):
            sx = isx +1
            sy = isy +1
	    blk_int = int_img[sy:,sx:] + int_img[0:-sy,0:-sx] - int_img[sy:,0:-sx] - int_img[0:-sy,sx:]
	    blk_dimy, blk_dimx = blk_int.shape
	    #fmap_dimx = blk_dimx - 2*sx
	    #fmap_dimy = blk_dimy - 2*sy
	    fmap_dimx = blk_dimx - 2
	    fmap_dimy = blk_dimy - 2
            fmap = np.zeros([fmap_dimy,fmap_dimx])
	    #coord = [[0,0],[0,sx],[0,2*sx],[sy,2*sx],[2*sy,2*sx],[2*sy,sx],[2*sy,0],[sy,0]]
	    coord = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]
	    #blk_center = blk_int[sy:sy+fmap_dimy,sx:sx+fmap_dimx]
	    blk_center = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
	    for ind in range(num_neighbours):
		fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= blk_center)
    return fmap




