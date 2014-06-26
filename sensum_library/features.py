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
sys.path.append("C:\\OSGeo4W64\\apps\\Python27\\Lib\\site-packages")
sys.path.append("C:\\OSGeo4W64\\apps\\orfeotoolbox\\python")
os.environ["PATH"] = os.environ["PATH"] + "C:\\OSGeo4W64\\bin"
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from conversion import *
from classification import *
import multiprocessing
import time


#os.chdir('F:\\Sensum_xp\\Izmir\\')

input_raster = 'building_extraction_sup\\pansharp.TIF'
input_shape = 'building_extraction_sup\\segmentation_watershed_default_training.shp'
input_raster_2 = 'wetransfer-749d73\\pansharp.TIF'
input_shape_2 = 'wetransfer-749d73\\watershed_001_training.shp'
input_raster_3 = 'F:\\Sensum_xp\\Izmir\\Landsat5\\1984\\LT51800331984164XXX04_B1_city_adj.TIF'
input_txt = 'building_extraction_sup\\svm.txt'
data_type = np.float32
output_raster_opencv = 'building_extraction_sup\\opencv_supervised.TIF'
output_raster_opencv_2 = 'wetransfer-749d73\\opencv_supervised.TIF'
output_raster_otb = 'building_extraction_sup\\otb_unsupervised.TIF'
output_raster_sup = 'wetransfer-749d73\\test_supervised_svm.TIF'
training_field = 'Class'

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
            NDISI = ((input_band_list[5] - ((input_band_list[0] + input_band_list[3] + input_band_list[4])/3)) / (input_band_list[5] + ((input_band_list[0] + input_band_list[3] + input_band_list[4])/3)))
            output_list.append(NDISI)
        if indexes_list[indx] == 'BUILT_UP' and len(input_band_list)>6:
            BUILT_UP = ((input_band_list[6]+input_band_list[1] - 1.5*input_band_list[4]) / (input_band_list[1] + input_band_list[4] + input_band_list[6]+0.0001))
            output_list.append(BUILT_UP)
            
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
    seg_pos = np.where(input_band_segmentation==dn_value)
    mat_pos = np.zeros(len(seg_pos[0]))
    if len(seg_pos[0]!=0):
        for l in range(0,len(seg_pos[0])):
            mat_pos[l] = input_band[seg_pos[0][l]][seg_pos[1][l]]
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
            for b in range(0,len(self.input_band_list)):
                data_glcm_1 = self.band_list_q[0][self.i:self.i+self.window_dimension,j:j+self.window_dimension] #extract the data for the glcm
            
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
        if (check):
            res = np.zeros(1)
        return res
    def __str__(self):
        return str(self.i)
    
    
if __name__ == '__main__':
    print time.asctime( time.localtime(time.time()) )
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    
    # Start consumers
    num_consumers = multiprocessing.cpu_count()  * 2
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
        
    
    input_band_list = read_image(input_raster_3,np.int32,0)
#    texture_moving_window(input_band_list,7,'dissimilarity',64) 
    
    window_dimension = 5
    index = 'dissimilarity'
    quantization_factor = 64
    
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
#    rows_w = 50
    
    for i in range(0,rows_w):
        tasks.put(Task_moving(i, rows_w, cols_w, input_band_list,band_list_q,window_dimension,index,quantization_factor))
        
        
     # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)
        #print tasks
    # Wait for all of the tasks to finish
    tasks.join()
    
    # Start printing results
    while rows_w:
        res = results.get()
        if res.size != 1:
            res = res.reshape(res.size/4,4)
            for i in range(res.size/4):
                tmp = res[i]
                b,index_row,index_col,feat1 = tmp[0],tmp[1],tmp[2],tmp[3]
                #print b,index_row,index_col,feat1
                output_ft_1[b][index_row][index_col]=float(feat1)
                #print output_ft_1[b][index_row][index_col]
        rows_w -= 1
    
    #print output_ft_1[b]
    
    for b in range(0,len(input_band_list)):
        output_list.append(output_ft_1[b][:][:])
        
    print time.asctime( time.localtime(time.time()) )
    print output_list

