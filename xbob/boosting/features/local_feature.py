"""The module implements the interfaces for block based local feature extraction methods.
The implemented features are Local Binary Pattern and its variants (tLbp, dLBP, mLBP). The features 
are extracted using blocks of different scale. Integral images are used to efficiently extract the 
features. """




import numpy


class lbp_feature():
    """ The class to extract block based LBP type features from the image.

    The class provides function to extract LBP and its variants (tLBP, dLBP, mLBP) features from the 
    image. These features are extracted from the blocks of the images and the size of the blocks can be varied.
    The number of neighbouring blocks are fixed to eight that correspond to the original LBP structure. """

    def __init__(self, ftype):
        """The function to initialize the feature type. 
 
        The function initializes the type of feature to be extracted. The type of feature can be one of the following
        lbp: The original LBP features that take difference of centre with the eight neighbours.
        tlbp: It take the difference of the neighbours with the adjacent neighbour and central values is ignored.
        dlbp: The difference between the pixels is taken along four different directions.
        mlbp: The difference of the neighbouring values is taken with the average of neighbours and the central value."""
        self.ftype = ftype
        self.coord = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]]


    def compute_integral_image(self,img):
        """The function computes an integral image for the given image.

        The function computes the integral image for the efficient computation of the block based features.
        Inputs:
        self: feature object
        img: Input images

        return:
        integral_xy: The integral image of the input image."""

        integral_y = numpy.cumsum(img,0)
        integral_xy = numpy.cumsum(integral_y,1)
        return integral_xy

    def get_features(self, img, scale_max_x, scale_max_y):
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
        feature_vector: The concatenated feature vectors for all the scales."""

        # Compute the integral image and pad zeros along row and col for block processing
        integral_imgc = self.compute_integral_image(img)
        rows, cols = img.shape
        integral_img = numpy.zeros([rows+1,cols+1])
        integral_img[1:,1:] = integral_imgc 

        # initialize
        num_neighbours = 8
        
        feature_vector = numpy.empty(0, dtype = 'uint8')

        # Vary the scale of the block and compute features
        for scale_x in range(scale_max_x):
            for scale_y in range(scale_max_y):

                # Compute the sum of the blocks for the current scale
                block_sum = integral_img[scale_y+1:,scale_x+1:] + integral_img[0:-(scale_y+1),0:-(scale_x+1)] - integral_img[scale_y+1:,0:-(scale_x+1)] - integral_img[0:-(scale_y+1),scale_x+1:]



                # extract the specific feature from the image
                if self.ftype == 'lbp':
                    feature_map = self.lbp(self.coord, feature_map_dimx, feature_map_dimy, block_sum)
                elif self.ftype == 'tlbp':
                    feature_map = self.tlbp(self.coord, feature_map_dimx, feature_map_dimy, block_sum)
                elif self.ftype == 'dlbp':
                    feature_map = self.dlbp(self.coord, feature_map_dimx, feature_map_dimy, block_sum)
                elif self.ftype == 'mlbp':
                    feature_map = self.mlbp(self.coord, feature_map_dimx, feature_map_dimy, block_sum)

                # reshape feature image into vector
                temp_vector = numpy.reshape(feature_map,feature_map.shape[0]*feature_map.shape[1],1)
 
                # concatenate the vector
                feature_vector = numpy.hstack((feature_vector,temp_vector))
        return feature_vector



    def lbp(self, block_sum):
        """Function to compute the LBP(3x3) for a image at single scale. 

        The LBP features of the given image is computed and the feature map is returned

        Inputs:
        block_sum: The image with each pixel representing the sum of a block.

        Return:
        feature_map: The lbp feature map
        """

        feature_map_dimx, feature_map_dimy = self.get_map_dimension(block_sum)
        num_neighbours = 8
        blk_center = block_sum[1:1+feature_map_dimy,1:1+feature_map_dimx]
        feature_map = numpy.zeros([feature_map_dimy, feature_map_dimx])
        for ind in range(num_neighbours):
            feature_map = feature_map + (2**ind)*(block_sum[self.coord[ind][0]:self.coord[ind][0] + feature_map_dimy,self.coord[ind][1]:self.coord[ind][1] + feature_map_dimx]>= blk_center)
        return feature_map



    def tlbp(self, block_sum):
        """Function to compute the tLBP for a image at single scale. 

        The tLBP features of the given image is computed and the feature map is returned

        Inputs:
        block_sum: The image with each pixel representing the sum of a block.

        Return:
        feature_map: The lbp feature map
        """
        feature_map_dimx, feature_map_dimy = self.get_map_dimension(block_sum)
        feature_map = numpy.zeros([feature_map_dimy, feature_map_dimx])
        num_neighbours = 8

        """ Compute the feature map for the tLBP features. """
        for ind in range(num_neighbours):
            
            """The comparison of pixel is done with the adjacent neighbours."""
            comparing_img = block_sum[self.coord[(ind+1)%num_neighbours][0]:self.coord[(ind+1)%num_neighbours][0] + feature_map_dimy,self.coord[(ind+1)%num_neighbours][1]:self.coord[(ind+1)%num_neighbours][1] + feature_map_dimx]
            
            """ Compare the neighbours and increment the feature map. """
            feature_map = feature_map + (2**ind)*(block_sum[self.coord[ind][0]:self.coord[ind][0] + feature_map_dimy,self.coord[ind][1]:self.coord[ind][1] + feature_map_dimx]>= comparing_img)
        return feature_map



    def dlbp(self, block_sum):
        """Function to compute the dLBP for a image at single scale. 

        The dLBP features of the given image is computed and the feature map is returned

        Inputs:
        block_sum: The image with each pixel representing the sum of a block.

        Return:
        feature_map: The lbp feature map
        """

        feature_map_dimx, feature_map_dimy = self.get_map_dimension(block_sum)
        pc = block_sum[1:1+feature_map_dimy,1:1+feature_map_dimx]
        num_neighbours = 8
        feature_map = numpy.zeros([feature_map_dimy,feature_map_dimx])
        for ind in range(num_neighbours/2):

            """The comparison of pixel is done with the diagonal neighbours."""
            pi = block_sum[self.coord[ind][0]:self.coord[ind][0]+ feature_map_dimy,self.coord[ind][1]:self.coord[ind][1] + feature_map_dimx]
            pi4 = block_sum[self.coord[ind+4][0]:self.coord[ind+4][0]+ feature_map_dimy,self.coord[ind+4][1]:self.coord[ind+4][1] + feature_map_dimx]

            """ Compare the neighbours and increment the feature map. """
            feature_map = feature_map + (2**(2*ind))*((pi-pc)*(pi4 - pc) >= 0) + (2**(2*ind+1))*(abs(pi - pc) >= abs(pi4 -pc))

        return feature_map



    def mlbp(self, block_sum):
        """Function to compute the mLBP for a image at single scale. 

        The mLBP features of the given image is computed and the feature map is returned. 

        Inputs:
        block_sum: The image with each pixel representing the sum of a block.

        Return:
        feature_map: The lbp feature map
        """

        feature_map_dimx, feature_map_dimy = self.get_map_dimension(block_sum)

        num_neighbours = 8
        pm = numpy.zeros([feature_map_dimy,feature_map_dimx])

        """The comparison of pixel is done with the average of the neighbours and central pixel."""
        for ind in range(num_neighbours):
            pm = pm + block_sum[self.coord[ind][0]:self.coord[ind][0]+ feature_map_dimy,self.coord[ind][1]:self.coord[ind][1] + feature_map_dimx]
        pm = pm/num_neighbours

        feature_map = numpy.zeros([feature_map_dimy,feature_map_dimx])
        for ind in range(num_neighbours):

            """ Select the value of the current neighbour.""" 
            pi = block_sum[self.coord[ind][0]:self.coord[ind][0]+ feature_map_dimy,self.coord[ind][1]:self.coord[ind][1] + feature_map_dimx]

            """ Compare the neighbours and increment the feature map. """
            feature_map = feature_map + (2**ind)*(pi >= pm)
        return feature_map
            
       
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

    def get_map_dimension(self, block_sum):
        """ The function computes the dimension of the LBP (R = 1, P = 8) type feature image

        The feature image returned by LBP with radius R is less than the image size by 2*R -1, for 
        R = 1, the feature map is smaller than the image by 2 pixel in both direction.

        Input: 

        block_sum: The image with each pixel representing the sum of a block.

        Return:
        feature_map_dimx, feature_map_dimy : The dimensions of the feature map."""

        # Initialize the size of the final feature map that will be obtained
        feature_map_dimy = block_sum.shape[0] -2    
        feature_map_dimx = block_sum.shape[1] -2
        return feature_map_dimx, feature_map_dimy

