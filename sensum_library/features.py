'''
.. module:: features
   :platform: Unix, Windows
   :synopsis: This module includes functions related to the extraction of image features from multi-spectral satellite images.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia
'''
'''
---------------------------------------------------------------------------------
                                features.py
---------------------------------------------------------------------------------
Created on May 13, 2013
Last modified on Mar 19, 2014
---------------------------------------------------------------------------------
Project: Framework to integrate Space-based and in-situ sENSing for dynamic 
         vUlnerability and recovery Monitoring (SENSUM)

Co-funded by the European Commission under FP7 (Seventh Framework Programme)
THEME [SPA.2012.1.1-04] Support to emergency response management
Grant agreement no: 312972

---------------------------------------------------------------------------------
License: This file is part of SensumTools.

    SensumTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SensumTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SensumTools.  If not, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------------------
'''
import config
import os
import numpy as np
import scipy.stats
import os,sys
import collections
from operator import itemgetter, attrgetter
#sys.path.append("C:\\OSGeo4W64\\apps\\Python27\\Lib\\site-packages")
#sys.path.append("C:\\OSGeo4W64\\apps\\orfeotoolbox\\python")
#os.environ["PATH"] = os.environ["PATH"] + "C:\\OSGeo4W64\\bin"
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from conversion import *
from classification import *
import multiprocessing
import time

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'


def band_calculation(input_band_list,indexes_list):
    
    '''Calculation of different indexes based on user selection
    
    :param input_band_list: list of 2darrays corresponding to bands (band 1: blue) (list of numpy arrays)
    :param indexes_list: list of strings with codes related to indexes (SAVI, NDVI, MNDWI, NDBI, NBI, NDISI, BUILT_UP) (list of strings)
    :returns list of 2darray corresponding to computed indexes following the indexes_list order (list of numpy arrays)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    ''' 
    
    output_list = []
    for indx in range(0,len(indexes_list)): #computes the index according to the input 
        if indexes_list[indx] == 'SAVI' and len(input_band_list)>3:
            SAVI = (((input_band_list[3] - input_band_list[2])*(8+0.5)) / (input_band_list[2] + input_band_list[3]+0.0001+0.5))
            output_list.append(SAVI)
        if indexes_list[indx] == 'NDVI' and len(input_band_list)>3:
            NDVI = (input_band_list[3]-input_band_list[2]) / (input_band_list[3]+input_band_list[2]+0.000001)
            output_list.append(NDVI)
        if indexes_list[indx] == 'MNDWI' and len(input_band_list)>4:
            MNDWI = ((input_band_list[1]-input_band_list[4]) / (input_band_list[1]+input_band_list[4]+0.0001))
            output_list.append(MNDWI)
        if indexes_list[indx] == 'NDBI' and len(input_band_list)>4:    
            NDBI = ((input_band_list[4] - input_band_list[3]) / (input_band_list[4] + input_band_list[3]+0.0001))
            output_list.append(NDBI)
        if indexes_list[indx] == 'NBI' and len(input_band_list)>4: 
            NBI = ((input_band_list[2] * input_band_list[4]) / (input_band_list[3]+0.0001))
            output_list.append(NBI)
        if indexes_list[indx] == 'NDISI' and len(input_band_list)>5:
            NDISI = ((input_band_list[5] - ((input_band_list[0] + input_band_list[3] + input_band_list[4])/3)) / (input_band_list[5] + ((input_band_list[0] + input_band_list[3] + input_band_list[4])/3))+0.00001)
            output_list.append(NDISI)
        if indexes_list[indx] == 'BUILT_UP':
            if len(input_band_list) < 9:
                BUILT_UP = ((input_band_list[6] + input_band_list[1] - 1.5*input_band_list[3]) / (input_band_list[1] + input_band_list[4] + input_band_list[6]+0.00001))
            else:
                BUILT_UP = ((input_band_list[7] + input_band_list[1] - 1.5*input_band_list[3]) / (input_band_list[1] + input_band_list[4] + input_band_list[6]+0.00001))
            output_list.append(BUILT_UP)
        '''    
        if indexes_list[indx] == 'Index5' and len(input_band_list)>6:
            if len(input_band_list) < 9:
                Index5 = (input_band_list[6]-input_band_list[1]) / (input_band_list[6]+input_band_list[1]+0.000001)
            else:
                Index5 = (input_band_list[7]-input_band_list[1]) / (input_band_list[7]+input_band_list[1]+0.000001)
            output_list.append(Index5)
        if indexes_list[indx] == 'Index4' and len(input_band_list)>6:
            Index4 = (input_band_list[4]-input_band_list[1]) / (input_band_list[4]+input_band_list[1]+0.000001)
            output_list.append(Index4)
        if indexes_list[indx] == 'Index3' and len(input_band_list)>6:
            Index3 = (input_band_list[4]-input_band_list[3]) / (input_band_list[4]+input_band_list[3]+0.000001)
            output_list.append(Index3)
        if indexes_list[indx] == 'Index2' and len(input_band_list)>6:
            Index2 = (input_band_list[3]-input_band_list[1]) / (input_band_list[3]+input_band_list[1]+0.000001)
            output_list.append(Index2)
        if indexes_list[indx] == 'Index1' and len(input_band_list)>6:
            if len(input_band_list) < 9:
                Index1 = (input_band_list[6]-input_band_list[4]) / (input_band_list[6]+input_band_list[4]+0.000001)
            else:
                Index1 = (input_band_list[7]-input_band_list[4]) / (input_band_list[7]+input_band_list[4]+0.000001)
            output_list.append(Index1)
        '''
        if indexes_list[indx] == 'Index1': #b4-b3/b4+b3
            index1 = (input_band_list[3] - input_band_list[2]) / (input_band_list[3] + input_band_list[2]+0.000001)
            output_list.append(index1)
        if indexes_list[indx] == 'Index2': #b5-b4/b5+b4
            index2 = (input_band_list[4] - input_band_list[3]) / (input_band_list[4] + input_band_list[3]+0.000001)
            output_list.append(index2)
        if indexes_list[indx] == 'Index3' and len(input_band_list) > 6: #b4-b7/b4+b7
            if len(input_band_list) < 9:
                index3 = (input_band_list[3] - input_band_list[6]) / (input_band_list[3] + input_band_list[6]+0.000001)
            else:
                index3 = (input_band_list[3] - input_band_list[7]) / (input_band_list[3] + input_band_list[7]+0.000001)
            output_list.append(index3)
        if indexes_list[indx] == 'Index4': #b5-b2/b5+b2
            index4 = (input_band_list[4] - input_band_list[1]) / (input_band_list[4] + input_band_list[1]+0.000001)
            output_list.append(index4)
        if indexes_list[indx] == 'Index5': #b5-b6/b5+b6
            index5 = (input_band_list[4] - input_band_list[5]) / (input_band_list[4] + input_band_list[5]+0.000001)
            output_list.append(index5)
        if indexes_list[indx] == 'Index6': #b7-b5/b7+b5
            if len(input_band_list) < 9:
                index6 = (input_band_list[6] - input_band_list[4]) / (input_band_list[6] + input_band_list[4]+0.000001)
            else:
                index6 = (input_band_list[7] - input_band_list[4]) / (input_band_list[7] + input_band_list[4]+0.000001)
            output_list.append(index6)
        if indexes_list[indx] == 'Index7': #b4-b2/b4+b2
            index7 = (input_band_list[3] - input_band_list[1]) / (input_band_list[3] + input_band_list[1]+0.000001)
            output_list.append(index7)
        if indexes_list[indx] == 'Index8': #b7-b2/b7+b2
            if len(input_band_list) < 9:
                index8 = (input_band_list[6] - input_band_list[1]) / (input_band_list[6] + input_band_list[1]+0.000001)
            else:
                index8 = (input_band_list[7] - input_band_list[1]) / (input_band_list[7] + input_band_list[1]+0.000001)
            output_list.append(index8) 
        if indexes_list[indx] == 'Index9': #
            #index A = float(BAND5) / (float(BAND5)+float(BAND4))
            inda = input_band_list[4].astype(float) / (input_band_list[4].astype(float)+input_band_list[3].astype(float)+0.000001)
            #index B = float(BAND4) / (float(BAND4)+float(BAND3))
            indb = input_band_list[3].astype(float) / (input_band_list[3].astype(float)+input_band_list[2].astype(float)+0.000001)
            #index C = float(BAND2) / (float(BAND2)+float(BAND5))
            indc = input_band_list[1].astype(float) / (input_band_list[1].astype(float)+input_band_list[4].astype(float)+0.000001)
            #index 9 = =(2.0*float(index A)-(float(index B)+float(index C)))/(2.0*float(index A)+(float(index B)+float(index C)))
            index9 = (2*inda.astype(float) - (indb.astype(float)+indc.astype(float))) / (2*inda.astype(float) + indb.astype(float)+indc.astype(float)+0.000001)
            output_list.append(index9) 
        if indexes_list[indx] == 'Index10': 
            #index 10 = (float(BAND5)-float(BAND4))/(10*sqrt(float(BAND5)+float(BAND6)))
            index10 = (input_band_list[4].astype(float) - input_band_list[3].astype(float)) / (10*np.sqrt(input_band_list[4].astype(float) + input_band_list[5].astype(float))+0.000001)
            output_list.append(index10)
        if indexes_list[indx] == 'Index11': 
            index11 = index1.astype(float) +0.5 / (np.absolute(index1.astype(float)+0.5)+0.000001) * np.sqrt(np.absolute(index1.astype(float)+0.5))
            output_list.append(index11)
        if indexes_list[indx] == 'Index12': 
            index1 = (input_band_list[3] - input_band_list[2]) / (input_band_list[3] + input_band_list[2]+0.000001)
            index12 = np.sqrt(np.absolute(index1.astype(float)+0.5))
            output_list.append(index12)
    return output_list
    
    
def pca(input_band_list):
    
    '''Principal Component Analysis
    
    :param input_band_list: list of 2darrays (list of numpy arrays)
    :returns:  a list containing mean, mode, second order component, third order component
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    #TODO: adjust description and input arguments. Again, a standard IO is needed for all functions.
    
    rows,cols = input_band_list[0].shape    
    #expand the listclass
    immatrix = np.array([np.array(input_band_list[i]).flatten() for i in range(0,len(input_band_list))],'f')
    
    #get dimensions
    num_data,dim = immatrix.shape

    #center data
    img_mean = immatrix.mean(axis=0)
    
    for i in range(num_data):
        immatrix[i] -= img_mean
    
    if dim>100:
        print 'PCA - compact trick used'
        M = np.dot(immatrix,immatrix.T) #covariance matrix
        e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = np.dot(immatrix.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
    else:
        print 'PCA - SVD used'
        U,S,V = np.linalg.svd(immatrix)
        V = V[:num_data] #only makes sense to return the first num_data    

    pca_mean = img_mean.reshape(rows,cols)
    pca_mode = V[0].reshape(rows,cols)
    pca_second_order = V[1].reshape(rows,cols)
    pca_third_order = V[2].reshape(rows,cols)       
    
    return pca_mean,pca_mode,pca_second_order,pca_third_order


def pca_index(pca_mean,pca_mode,pca_sec_order,pca_third_order):
    
    '''PCA-based index for built-up area extraction
    
    :param pca_mean: matrix with mean computed by pca (numpy array)
    :param pca_mode: matrix with mode computed by pca (numpy array)
    :param pca_second_order: matrix with second order component computed by pca (numpy array)
    :param pca_third_order: matrix with third order component computed by pca (numpy array)
    :returns:  a matrix with the pca built-up indicator
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    pca_built_up = ((4*pca_mode)+pca_mean)/(pca_mean + pca_mode+pca_sec_order+pca_third_order+0.0001)
    return pca_built_up


def spectral_segments(input_band,dn_value,input_band_segmentation,indexes_list,bands_number):
    
    '''Compute the desired spectral features from each segment
    
    :param input_band: 2darray containing a single band of the original image (numpy array)
    :param dn_value: unique value associated to each segment (integer)
    :param input_band_segmentation: 2darray containing the results of the segmentation (numpy array)
    :param indexes_list: list of strings with codes related to indexes (mean, mode, std, max_br, min_br, ndvi_mean, ndvi_std, weigh_br) (list of strings)
    :param bands_number: parameter used by the weighted brightness (set to 0 if not needed) (integer)
    :returns:  list of values corresponding to computed indexes following the indexes_list order (list of floats)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    #TODO: What is the form of the output? - A 2d matrix with one stats value per segment?
    #TODO: So one has to run it per band or does it take a nd matrix as input and outputs an nd matrix? -> alternative could be to use OpenCV
    
    output_list = []
    mean = 0.0
    std = 0.0
    mode = 0.0
    maxbr = 0.0
    minbr = 0.0
    weigh_br = 0.0
    
    mask = np.equal(input_band_segmentation,dn_value)
    data = np.extract(mask,input_band)
    mat_pos = data.flatten()
    #seg_pos = np.where(input_band_segmentation==dn_value)
    #mat_pos = np.zeros(len(seg_pos[0]))
    #if len(seg_pos[0]!=0):
        #for l in range(0,len(seg_pos[0])):
         #   mat_pos[l] = input_band[seg_pos[0][l]][seg_pos[1][l]]
    for indx in range(0,len(indexes_list)):
        if indexes_list[indx] == 'mean' or indexes_list[indx] == 'ndvi_mean':
            mean = mat_pos.mean()
            output_list.append(mean)
        if indexes_list[indx] == 'std' or indexes_list[indx] == 'ndvi_std':
            std = mat_pos.std()
            output_list.append(std)
        if indexes_list[indx] == 'mode':
            mode_ar = scipy.stats.mode(mat_pos)
            mode = mode_ar[0][0]
            output_list.append(mode)
        if indexes_list[indx] == 'max_br':
            maxbr = np.amax(mat_pos)
            output_list.append(maxbr)
        if indexes_list[indx] == 'min_br':
            minbr = np.amin(mat_pos)
            output_list.append(minbr)
        if indexes_list[indx] == 'weigh_br':
            npixels = np.sum(mask)
            outmask_band_sum = np.choose(mask,(0,input_band)) 
            values = np.sum(outmask_band_sum)
            nbp = bands_number*npixels
            div = 1.0/nbp
            weigh_br = div*values
            output_list.append(weigh_br)
    
    mat_pos=None
    seg_pos=None
    mask=None
    return output_list


def texture_segments(input_band,dn_value,input_band_segmentation,indexes_list):
    
    '''Compute the desired spectral features from each segment
    
    :param input_band: 2darray containing a single band of the original image (numpy array, unsigned integer 8bit)
    :param dn_value: unique value associated to each segment (integer)
    :param input_band_segmentation: 2darray containing the results of the segmentation (numpy array)
    :param indexes_list: list of strings with codes related to indexes (contrast, energy, homogeneity, correlation, dissimilarity, ASM) (list of strings)
    :returns:  list of values corresponding to computed indexes following the indexes_list order (list of floats)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    #TODO: can we go through this one together? do you calculate the glcm segment wise and output a 2d matrix with values per segment?
    
    index_glcm = 0.0
    output_list = []
    mask = np.equal(input_band_segmentation,dn_value)
    seg_pos = np.where(input_band_segmentation==dn_value)

    xstart = np.amin(seg_pos[1])
    xend = np.amax(seg_pos[1])

    ystart = np.amin(seg_pos[0])
    yend = np.amax(seg_pos[0])
    #data_glcm = np.zeros((yend-ystart+1,xend-xstart+1))
    
    data_glcm = input_band[ystart:yend+1,xstart:xend+1] #TODO: is this redefinition intended?
    
    glcm = greycomatrix(data_glcm, [1], [0], levels=64, symmetric=False, normed=True)
    for indx in range(0,len(indexes_list)):
        index_glcm = greycoprops(glcm, indexes_list[indx])[0][0]
        output_list.append(index_glcm)    
    
    return output_list
 
 #TODO

def texture_moving_window(input_band_list,window_dimension,index,quantization_factor):
    
    '''Compute the desired spectral feature from each window
    
    :param input_band_list: list of 2darrays (list of numpy arrays)
    :param window_dimension: dimension of the processing window (integer)
    :param index: string with index to compute (contrast, energy, homogeneity, correlation, dissimilarity, ASM) (string)
    :param quantization_factor: number of levels to consider (suggested 64) (integer)
    :returns:  list of 2darrays corresponding to computed index per-band (list of numpy arrays)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    #TODO: Please explain better what this function does. I assume it calculates GLCM derived features from a moving window.
    #TODO: Always provide full list of options in function description (e.g. which features are supported here?)
    #TODO: Output should be array. Only dissimilarity and only 3 bands? 
    
    band_list_q = linear_quantization(input_band_list,quantization_factor)
    output_list = []
    
               
    feat1 = 0.0
    
    rows,cols=input_band_list[0].shape
    output_ft_1 = np.zeros((len(input_band_list),rows,cols)).astype(np.float32)
    
    print input_band_list[0].shape
    if (rows%window_dimension)!=0:
        rows_w = rows-1
    else:
        rows_w = rows
    if (cols%window_dimension)!=0:
        cols_w = cols-1
    else:
        cols_w = cols
    print rows,cols
#
#    rows_w = 10
    for i in range(0,rows_w):
        print str(i+1)+' of '+str(rows_w)
        for j in range(0,cols_w):
            for b in range(0,len(input_band_list)):
                data_glcm_1 = band_list_q[0][i:i+window_dimension,j:j+window_dimension] #extract the data for the glcm
            
                if (i+window_dimension<rows_w) and (j+window_dimension<cols_w):
                    glcm1 = greycomatrix(data_glcm_1, [1], [0, np.pi/4, np.pi/2, np.pi*(3/4)], levels=quantization_factor, symmetric=False, normed=True)
                    feat1 = greycoprops(glcm1, index)[0][0]
                    index_row = i+1 #window moving step
                    index_col = j+1 #window moving step
                    
                    output_ft_1[b][index_row][index_col]=float(feat1) #stack to store the results for different bands
    for b in range(0,len(input_band_list)):
        output_list.append(output_ft_1[b][:][:])
    
    return output_list

    
class Task_moving(object):
    #import os,sys
    #sys.path.append("C:\\OSGeo4W64\\apps\\Python27\\Lib\\site-packages")
    #sys.path.append("C:\\OSGeo4W64\\apps\\orfeotoolbox\\python")
    #os.environ["PATH"] = os.environ["PATH"] + "C:\\OSGeo4W64\\bin"
    def __init__(self, i, rows_w, cols_w, input_band_list,band_list_q,window_dimension,index, quantization_factor):
        self.i = i
        self.rows_w = rows_w
        self.cols_w = cols_w
        self.input_band_list = input_band_list
        self.band_list_q = band_list_q
        self.window_dimension = window_dimension
        self.index = index
        self.quantization_factor = quantization_factor
    def __call__(self):
        global res
        check = 1
        print str(self.i+1)+' of '+str(self.rows_w)
        for j in range(0,self.cols_w):
            data_glcm_1 = self.band_list_q[0][self.i:self.i+self.window_dimension,j:j+self.window_dimension] #extract the data for the glcm
            data_glcm_2 = self.band_list_q[1][self.i:self.i+self.window_dimension,j:j+self.window_dimension] #extract the data for the glcm
            data_glcm_3 = self.band_list_q[2][self.i:self.i+self.window_dimension,j:j+self.window_dimension] #extract the data for the glcm
            if (self.i+self.window_dimension<self.rows_w) and (j+self.window_dimension<self.cols_w):
                glcm1 = greycomatrix(data_glcm_1, [1], [0, np.pi/4, np.pi/2, np.pi*(3/4)], levels=self.quantization_factor, symmetric=False, normed=True)
                feat1 = greycoprops(glcm1, self.index)[0][0]
                glcm2 = greycomatrix(data_glcm_2, [1], [0, np.pi/4, np.pi/2, np.pi*(3/4)], levels=self.quantization_factor, symmetric=False, normed=True)
                feat2 = greycoprops(glcm2, self.index)[0][0]
                glcm3 = greycomatrix(data_glcm_3, [1], [0, np.pi/4, np.pi/2, np.pi*(3/4)], levels=self.quantization_factor, symmetric=False, normed=True)
                feat3 = greycoprops(glcm3, self.index)[0][0]
                index_row = self.i+1
                index_col = j+1
                if (check):
                    res = []
                    check = 0
                tmp1 = np.array([0,index_row,index_col,feat1])
                tmp2 = np.array([1,index_row,index_col,feat2])
                tmp3 = np.array([2,index_row,index_col,feat3])
                res = np.append(res,tmp1)
                res = np.append(res,tmp2)
                res = np.append(res,tmp3)
            '''
            for b in range(0,len(self.input_band_list)):
                data_glcm_1 = self.band_list_q[b][self.i:self.i+self.window_dimension,j:j+self.window_dimension] #extract the data for the glcm
                if (self.i+self.window_dimension<self.rows_w) and (j+self.window_dimension<self.cols_w):
                    glcm1 = greycomatrix(data_glcm_1, [1], [0, np.pi/4, np.pi/2, np.pi*(3/4)], levels=self.quantization_factor, symmetric=False, normed=True)
                    feat1 = greycoprops(glcm1, self.index)[0][0]
                    index_row = self.i+1 #window moving step
                    index_col = j+1 #window moving step
                    #FIX IT NOOB
                    if (check):
                        res = []
                        check = 0
                    tmp = np.array([b,index_row,index_col,feat1])
                    res = np.append(res,tmp)
            '''
        if (check):
            res = np.zeros(1)
        return res
    def __str__(self):
         return str(self.i)
    

def value_to_segments(input_raster,input_shape,output_shape,operation = 'Mode'):
    
    '''Assign the most frequent value inside a segment to the segment itself
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param input_shape: path and name of shapefile with the segmentation results (*.shp) (string)
    :param output_shape: path and name of the output shapefile (*.shp) (string)
    :param operation: string with the function to apply to fill the segments (mean or mode) (string)
    :returns:  an output shapefile is created with a new attribute field related to the most frequent value inside the segment
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 23/03/2014
    '''
    #Example of hybrid approach
    #TODO: this is only a spatial union operation, isn't it? So it is not part of the hybrid approach where you aggregate pixel classes to segments!?
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster) 
    band_list_class = read_image(input_raster,np.int32,0) #read original raster file
    shp2rast(input_shape,input_shape[:-4]+'_conv.TIF',rows,cols,'DN',0,0,0,0,0,0) #conversion of the segmentation results from shape to raster for further processing
    band_list_seg = read_image(input_shape[:-4]+'_conv.TIF',np.int32,0) #read segmentation raster file
    
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile=driver_shape.Open(input_shape)
    inlayer=infile.GetLayer()
    outfile=driver_shape.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Features',geom_type=osgeo.ogr.wkbPolygon)
    
    layer_defn = inlayer.GetLayerDefn()
    infeature = inlayer.GetNextFeature()
    feature_def = outlayer.GetLayerDefn()
    dn_def = osgeo.ogr.FieldDefn('DN', osgeo.ogr.OFTInteger)
    outlayer.CreateField(dn_def)
    for b in range(0,len(band_list_class)):
        class_def = osgeo.ogr.FieldDefn(operation+str(b+1),osgeo.ogr.OFTReal)
        outlayer.CreateField(class_def)
    area_def = osgeo.ogr.FieldDefn('Area',osgeo.ogr.OFTReal)
    outlayer.CreateField(area_def)
    
    n_feature = inlayer.GetFeatureCount()
    j = 1
    while infeature:
        if j%50 == 0:
            print str(j) + ' of ' + str(n_feature)
        j = j+1
        dn = infeature.GetField('DN')
        # get the input geometry
        geom = infeature.GetGeometryRef()
        area = geom.Area()
        # create a new feature
        outfeature = osgeo.ogr.Feature(feature_def)
        # set the geometry and attribute
        outfeature.SetGeometry(geom)
        #seg_pos = np.where(band_list_seg[0] == dn) #returns a list of x and y coordinates related to the pixels satisfying the given condition
        #mat_pos = np.zeros(len(seg_pos[0]))
        mask = np.equal(band_list_seg[0],dn)
        
        #Extract all the pixels inside a segment
        for b in range(0,len(band_list_class)):
            mat_pos = np.extract(mask,band_list_class[b])
            #for l in range(0,len(seg_pos[0])):
                #mat_pos[l] = band_list_class[b][seg_pos[0][l]][seg_pos[1][l]]
            if operation == 'Mode':
                mode_ar = scipy.stats.mode(mat_pos)
                value = mode_ar[0][0]
            else:
                value = np.mean(mat_pos)
            outfeature.SetField(operation+str(b+1),value)
            
        outfeature.SetField('DN',dn)
        outfeature.SetField('Area',area)
        outlayer.CreateFeature(outfeature)
        outfeature.Destroy() 
        infeature = inlayer.GetNextFeature()
    
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()    
    
    shutil.copyfile(input_shape[:-4]+'.prj', output_shape[:-4]+'.prj') #projection definition
    
    
def get_class(target_layer):
    
    urban_classes_tmp = []
    dissimilarity1_sums = []
    dissimilarity2_sums = []
    dissimilarity3_sums = []
    counters = []
    for i in range(target_layer.GetFeatureCount()):
        urban_class = target_layer.GetFeature(i).GetField("Class")
        dissimilarity1 = target_layer.GetFeature(i).GetField("Mean1")
        dissimilarity2 = target_layer.GetFeature(i).GetField("Mean2")
        dissimilarity3 = target_layer.GetFeature(i).GetField("Mean3")
        if urban_class not in urban_classes_tmp:
            urban_classes_tmp.append(urban_class)
            counters.append(1)
            dissimilarity1_sums.append(dissimilarity1)
            dissimilarity2_sums.append(dissimilarity2)
            dissimilarity3_sums.append(dissimilarity3)
        else:
            index = (i for i,urban_class_tmp in enumerate(urban_classes_tmp) if urban_class_tmp == urban_class).next()
            counters[index] = counters[index] + 1
            dissimilarity1_sums[index] = dissimilarity1_sums[index] + dissimilarity1
            dissimilarity2_sums[index] = dissimilarity2_sums[index] + dissimilarity2
            dissimilarity3_sums[index] = dissimilarity3_sums[index] + dissimilarity3
    for i,urban_class in enumerate(urban_classes_tmp):
        dissimilarity1_sums[i] = dissimilarity1_sums[i]/counters[i]
        dissimilarity2_sums[i] = dissimilarity2_sums[i]/counters[i]
        dissimilarity3_sums[i] = dissimilarity3_sums[i]/counters[i]
    index1 = (urban_classes_tmp[i] for i,disimmilarity1_sum in enumerate(dissimilarity1_sums) if disimmilarity1_sum == max(dissimilarity1_sums)).next()
    index2 = (urban_classes_tmp[i] for i,disimmilarity2_sum in enumerate(dissimilarity2_sums) if disimmilarity2_sum == max(dissimilarity2_sums)).next()
    index3 = (urban_classes_tmp[i] for i,disimmilarity3_sum in enumerate(dissimilarity3_sums) if disimmilarity3_sum == max(dissimilarity3_sums)).next()
    index_list = [index1,index2,index3]
    return max(set(index_list), key=index_list.count)


def tile_statistics(band_mat,start_col_coord,start_row_coord,end_col_coord,end_row_coord):

    '''
    Compute statistics related to the input tile

    :param band_mat: numpy 8 bit array containing the extracted tile
    :param start_col_coord: starting longitude coordinate
    :param start_row_coord: starting latitude coordinate
    :param end_col_coord: ending longitude coordinate
    :param end_row_coord: ending latitude coordinate

    :returns: a list of statistics (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence, min frequency value, max frequency value, standard deviation value, distance among frequent values)

    Author: Daniele De Vecchi
    Last modified: 22/08/2014
    '''

    #Histogram definition
    data_flat = band_mat.flatten()
    data_counter = collections.Counter(data_flat)
    data_common = (data_counter.most_common(20)) #20 most common values
    data_common_sorted = sorted(data_common,key=itemgetter(0)) #reverse=True for inverse order
    hist_value = [elt for elt,count in data_common_sorted]
    hist_count = [count for elt,count in data_common_sorted]

    #Define the level of confidence according to the computed statistics 
    min_value = hist_value[0]
    max_value = hist_value[-1]
    std_value = np.std(hist_count)
    diff_value = max_value - min_value
    min_value_count = hist_count[0]
    max_value_count = hist_count[-1] 
    tot_count = np.sum(hist_count)
    min_value_freq = (float(min_value_count) / float(tot_count)) * 100
    max_value_freq = (float(max_value_count) / float(tot_count)) * 100

    if max_value_freq > 20.0 or min_value_freq > 20.0 or diff_value < 18 or std_value > 100000:
        confidence = 0
    elif max_value_freq > 5.0: #or std_value < 5.5: #or min_value_freq > 5.0:
        confidence = 0.5
    else:
        confidence = 1

    return (start_col_coord,start_row_coord,end_col_coord,end_row_coord,confidence,min_value_freq,max_value_freq,std_value,diff_value)


def classification_statistics(input_raster_classification,input_raster):

    '''
    Compute statistics related to the input unsupervised classification

    :param input_raster_classification: path and name of the input raster file with classification(*.TIF,*.tiff) (string)
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)

    :returns: a list of statistics (value/class,min_value,max_value,diff_value,std_value,min_value_freq,max_value_freq,tot_count)

    Author: Daniele De Vecchi
    Last modified: 25/08/2014
    '''

    band_list_classification = read_image(input_raster_classification,np.uint8,0)
    rows_class,cols_class,nbands_class,geotransform_class,projection_class = read_image_parameters(input_raster_classification)

    band_list = read_image(input_raster,np.uint8,0)
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)

    max_class = np.max(band_list_classification[0])
    stat_list = []
    for value in range(0,max_class+1):
        #print '----------------------------'
        #print 'Class ' + str(value)
        mask = np.equal(band_list_classification[0],value)
        data = np.extract(mask,band_list[0])

        #Statistics
        #Histogram definition
        data_flat = data.flatten()
        data_counter = collections.Counter(data_flat)
        data_common = (data_counter.most_common(20)) #20 most common values
        data_common_sorted = sorted(data_common,key=itemgetter(0)) #reverse=True for inverse order
        hist_value = [elt for elt,count in data_common_sorted]
        hist_count = [count for elt,count in data_common_sorted]

        #Define the level of confidence according to the computed statistics 
        min_value = hist_value[0]
        max_value = hist_value[-1]
        std_value = np.std(hist_count)
        diff_value = max_value - min_value
        min_value_count = hist_count[0]
        max_value_count = hist_count[-1] 
        tot_count = np.sum(hist_count)
        min_value_freq = (float(min_value_count) / float(tot_count)) * 100
        max_value_freq = (float(max_value_count) / float(tot_count)) * 100

        #print 'Min value: ' + str(min_value)
        #print 'Max value: ' + str(max_value)
        #print 'Diff value: ' + str(diff_value)
        #print 'Standard Deviation: ' + str(std_value)
        #print 'Min value frequency: ' + str(min_value_freq)
        #print 'Max value frequency: ' + str(max_value_freq)
        #print 'Total values: ' + str(tot_count)
        #print '----------------------------'
        stat_list.append((value,min_value,max_value,diff_value,std_value,min_value_freq,max_value_freq,tot_count))
    return stat_list

