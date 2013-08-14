"""The module implements provide the interface for block based local feature extraction methods.
The features implemented are Local Binary Pattern and its variants (tLbp, dLBP, mLBP). The features 
are extracted using blocks of different scale. Integral images are used to effeciently extract the 
features. """




import numpy
import math


class lbp_feature():
    """ The class to extract block based LBP type features from the image.

    The class provides function to extract LBP and its variants (tLBP, dLBP, mLBP) features from the 
    image. These features are extracted from the blocks of the images and the size of the blocks can be varied.
    The number of neighbouring blocks are fixed to eight that correspond to the original LBP structure. """

    def __init__(self, ftype):
        """The function to initilize the feature type. 
 
        The function initilizes the type of feature to be extracted. The type of feature can be one of the following
        lbp: The original LBP features that take difference of center with the eight neighbours.
        tlbp: It take the difference of the neighbours with the adjacent neighbour and central values is ignored.
        dlbp: The difference between the pixels is taken along four different directions.
        mlbp: The difference of the neighbouring values is taken with the average of neighbours and the central value."""
        self.ftype = ftype


    def integral_img(self,img):
        """The function cumputes an intergal image for the given image.

        The function computes the intergral image for the effecient computation of the block based features.
        Inouts:
        self: feature object
        img: Input images

        return:
        int_img: The intergal image of the input image."""

        integral_x = numpy.cumsum(img,0)
        integral_img = numpy.cumsum(integral_x,1)
        return integral_img

    def get_features(self,img,cx,cy):
        """The function computes the block based local features at different scales.

        The function extracts the block based local features at multiple scales from the image. The scale refers to
        the size of the block used while computing the features. The feature captured at different scales are 
        concatenated together and the final feature vector is returned. The scale are varied from the lowest scale of 1
        to maximum which is provided as the parameter to the function (cx, cy). Thus the features are extract with the 
        block size of {1,2,3...cy} x {1,2,3...cx}.

        Inputs:
        img: Image for extracting the features.
        cx: The maximum columns for the block
        cy: The maximum rows for the block

        Return:
        fvec: The concatenated feature vectors for all the scales."""
        # Compute the intergal image and add zeros along row and col for block processing
        int_imgc = self.integral_img(img)
        rows, cols = img.shape
        int_img = numpy.zeros([rows+1,cols+1])
        int_img[1:,1:] = int_imgc 

        # initialize
        num_neighbours = 8
        coord = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]
        fvec = numpy.empty(0, dtype = 'uint8')

        # Vary the scale of the block and compute features
        for xi in range(cx):
            for yi in range(cy):
                # Compute the sum of the blocks for the current scale
                blk_int = int_img[yi+1:,xi+1:] + int_img[0:-(yi+1),0:-(xi+1)] - int_img[yi+1:,0:-(xi+1)] - int_img[0:-(yi+1),xi+1:]

                # Initialize the size of the final feature map that will be obtained
                fmap_dimy = blk_int.shape[0] -2    
                fmap_dimx = blk_int.shape[1] -2

                # extract the specific feature from the image
                if(self.ftype == 'lbp'):
                    fmap = self.lbp(coord, fmap_dimx, fmap_dimy, blk_int)
                elif(self.ftype == 'tlbp'):
                    fmap = self.tlbp(coord, fmap_dimx, fmap_dimy, blk_int)
                elif(self.ftype == 'dlbp'):
                    fmap = self.dlbp(coord, fmap_dimx, fmap_dimy, blk_int)
                elif(self.ftype == 'mlbp'):
                    fmap = self.mlbp(coord, fmap_dimx, fmap_dimy, blk_int)

                # reshape feature image into vector
                vec = numpy.reshape(fmap,fmap.shape[0]*fmap.shape[1],1)
 
                # concatenate the vector
                fvec = numpy.hstack((fvec,vec))
        return fvec



    def lbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        """Function to compute the LBP for a image at single scale. 

        The LBP features of the given image is computed and the feature map is returned

        Inputs:
        coord: The coordinates specify the neighbour to be considered.
        fmap_dimx: feature map's dimension along the columns.
        fmap_dimy: Feature maps dimension along the rows.

        Return:
        fmap: The lbp feature map
        """
        num_neighbours = 8
        blk_center = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
        fmap = numpy.zeros([fmap_dimy, fmap_dimx])
        for ind in range(num_neighbours):
            fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= blk_center)
        return fmap



    def tlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        """Function to compute the tLBP for a image at single scale. 

        The LBP features of the given image is computed and the feature map is returned

        Inputs:
        coord: The coordinates specify the neighbour to be considered.
        fmap_dimx: feature map's dimension along the columns.
        fmap_dimy: Feature maps dimension along the rows.

        Return:
        fmap: The lbp feature map
        """

        fmap = numpy.zeros([fmap_dimy, fmap_dimx])
        num_neighbour = 8

        for ind in range(num_neighbours):
            comp_img = blk_int[coord[(ind+1)%num_neighbour][0]:coord[(ind+1)%num_neighbour][0] + fmap_dimy,coord[(ind+1)%num_neighbour][1]:coord[(ind+1)%num_neighbour][1] + fmap_dimx]
            fmap = fmap + (2**ind)*(blk_int[coord[ind][0]:coord[ind][0] + fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]>= comp_img)
        return fmap



    def dlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        """Function to compute the dLBP for a image at single scale. 

        The LBP features of the given image is computed and the feature map is returned

        Inputs:
        coord: The coordinates specify the neighbour to be considered.
        fmap_dimx: feature map's dimension along the columns.
        fmap_dimy: Feature maps dimension along the rows.

        Return:
        fmap: The lbp feature map
        """

        pc = blk_int[1:1+fmap_dimy,1:1+fmap_dimx]
        num_neighbours = 8
        fmap = numpy.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours/2):
            pi = blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
            pi4 = blk_int[coord[ind+4][0]:coord[ind+4][0]+ fmap_dimy,coord[ind+4][1]:coord[ind+4][1] + fmap_dimx]
            fmap = fmap + (2**ind)*((pi-pc)*(pi4 - pc) > 0) + (4**ind)*(abs(pi - pc) >= abs(pi4 -pc))
        return fmap



    def mlbp(self, coord, fmap_dimx, fmap_dimy, blk_int):
        """Function to compute the mLBP for a image at single scale. 

        The LBP features of the given image is computed and the feature map is returned

        Inputs:
        coord: The coordinates specify the neighbour to be considered.
        fmap_dimx: feature map's dimension along the columns.
        fmap_dimy: Feature maps dimension along the rows.

        Return:
        fmap: The lbp feature map
        """

        num_neighbours = 8
        pm = numpy.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours):
            pm = pm + blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
        pm = pm/num_neighbours

        fmap = numpy.zeros([fmap_dimy,fmap_dimx])
        for ind in range(num_neighbours):
            pi = blk_int[coord[ind][0]:coord[ind][0]+ fmap_dimy,coord[ind][1]:coord[ind][1] + fmap_dimx]
            fmap = fmap + (2**ind)*(pi >= pm)
        return fmap
            
       
    def get_feature_number(self, dimy, dimx, scale_y, scale_x):
        """The function gives the feature size for given size of image and scales

        The number of features for the given parameters are computed

        Inputs:
        dimy: Number of rows of the image
        dimx: Number of columns of the image
        scale_y: The maximum block size along the rows
        scale_x: The maximum block size along the columns.

        Return:
        The total number of features obtained for these parameters."""
        img = numpy.zeros([dimy, dimx])
        feature_vector = self.get_features(img, scale_y, scale_x)
        return feature_vector.shape[0]


