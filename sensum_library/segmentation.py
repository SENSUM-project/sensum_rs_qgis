'''
.. module:: segmentation
   :platform: Unix, Windows
   :synopsis: This module includes functions related to the low-level segmentation of multi-spectral satellite images.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia
'''
'''
---------------------------------------------------------------------------------
                                segmentation.py
---------------------------------------------------------------------------------
Created on May 13, 2013
Last modified on Mar 22, 2014
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
import sys
import osgeo.gdal
from gdalconst import *
import numpy as np
import otbApplication
from skimage.segmentation import felzenszwalb, quickshift
from scipy import optimize
import random
import shutil
import glob
import collections
from operator import itemgetter
from conversion import *

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'

temp_folder_interimage = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+"/terraAIDA/.tmp"
exe_folder_interimage = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+"/terraAIDA"


def felzenszwalb_skimage(input_band_list, scale, sigma, min_size):
   
    '''Felzenszwalb segmentation from Skimage library
    
    :param input_band_list: list of 2darrays (list of numpy arrays)
    :param scale: defines the observation level, higher scale means less and larger segments (float)
    :param sigma: idth of Gaussian smoothing kernel for preprocessing, zero means no smoothing (float)
    :param min_size: minimum size, minimum component size. Enforced using postprocessing. (integer)
    :returns:  2darray with the result of the segmentation (numpy array)
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    ''' 
    
    #TODO: Also here seems to me that only RGB is used and each band is segmented separately and than merged.
    #TODO: Would be more powerful if full spectral content would be used and one segmentation is run on the n-D feature space.
    
    #default values, set in case of 0 as input
    if scale == 0:
        scale = 2
    if sigma == 0:
        sigma = 0.1
    if min_size == 0:
        min_size = 2
    
    if len(input_band_list) == 4:
        img = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2],input_band_list[3]))
    if len(input_band_list) == 3:
        img = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2]))
    if len(input_band_list) == 1:
        img = input_band_list[0]
    segments_fz = felzenszwalb(img, scale, sigma, min_size)

    return segments_fz


def quickshift_skimage(input_band_list,kernel_size, max_distance, ratio):
    
    '''Quickshift segmentation from Skimage library, works with RGB
    
    :param input_band_list: list of 2darrays (list of numpy arrays)
    :param kernel_size: width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters. (float)
    :param max_distance:cut-off point for data distances. Higher means fewer clusters. (float)
    :param ratio: balances color-space proximity and image-space proximity. Higher values give more weight to color-space. (float between 0 and 1)
    :returns:  2darray with the result of the segmentation (numpy array)
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift
    ''' 
    
    #TODO: Would add here directly also the related publication reference (for all functions that are based on scientific papers)
    
    #default values, set in case of 0 as input
    if kernel_size == 0:
        kernel_size = 5
    if max_distance == 0:
        max_distance = 10
    if ratio == 0:
        ratio = 1
    
    img = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2]))
    segments_quick = quickshift(img, kernel_size, max_distance, ratio)

    return segments_quick


def baatz_interimage(input_raster,euc_threshold,compactness,baatz_color,scale,index):#,input_bands,input_weights,output folder,reliability):
    
    '''Baatz segmentation from InterImage/TerraAIDA library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param euc_threshold: euclidean distance threshold. The minimum Euclidean Distance between each segment feature. (float, positive)
    :param compactness: Baatz Compactness Weight attribute (float, between 0 and 1)
    :param baatz_color: Baatz Color Weight attribute (float, between 0 and 1)
    :param scale: Baatz scale attribute (float, positive)
    :param index: progressive index in case of multiple segmentations to differentiate temp files (integer)
    :returns:  2darray with the result of the segmentation (numpy array)
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://interimage.sourceforge.net/
               http://www.dpi.inpe.br/terraaida/
               http://www.ecognition.cc/download/baatz_schaepe.pdf
    '''
    
    #TODO: Would exclude this function from the package as it is not fully open-source and needs to call an exe file.
    
    #default values, set in case of 0 as input
    if euc_threshold == 0:
        euc_threshold = 50
    if compactness  == 0:
        compactness = 0.5
    if baatz_color  == 0:
        baatz_color = 0.5
    if scale  == 0:
        scale = 80
    
    f = None

    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    
    if nbands > 1:
        bands_str = ','.join(str(i) for i in xrange(nbands))
        weights_str = ','.join('1' for i in xrange(nbands))
    else:
        bands_str = '0'
        weights_str = '1'
    
    GW=geo_transform[0]
    GN=geo_transform[3]
    a= pixel2world(geo_transform, cols,rows)
    #print a[0], a[1]
    GE= a[0]
    GS= a[1]
    
    output_file = temp_folder_interimage + separator +'baatz_' +  str(euc_threshold) + '_' + str(compactness) + '_' + str(baatz_color) + '_' + str(scale) + '_' + str(index)
    
    #removing the file created by the segmenter after each run
    Folder_files = os.listdir(temp_folder_interimage)
    #file_ = [s for s in Folder_files if "ta_segmenter" in s or "baatz" in s or "regiongrowing" in s]
    file_ = [s for s in Folder_files if "ta_segmenter" in s]
    for f in file_:
        os.remove(temp_folder_interimage+separator+f)
    #exe_file =  exe_folder_interimage +separator+ 'ta_baatz_segmenter.exe'   
    exe_file = (exe_folder_interimage +separator+"ta_baatz_segmenter.op" if os.name == "posix" else exe_folder_interimage +separator+"ta_baatz_segmenter.exe")
    #runs the baatz segmenter
    print exe_file + ' "'+input_raster+'" "'+str(GW)+'" "'+str(GN)+'" "'+str(GE)+'" "'+str(GS)+'" "" "'+temp_folder_interimage+'" "" Baatz "'+str(euc_threshold)+'" "@area_min@" "'+str(compactness)+'" "'+str(baatz_color)+'" "'+str(scale)+'" "'+str(bands_str)+ '" "' + str(weights_str)+'" "'+output_file+'" "seg" "0.2" "" "" "no"'
    os.system(exe_file + ' "'+input_raster+'" "'+str(GW)+'" "'+str(GN)+'" "'+str(GE)+'" "'+str(GS)+'" "" "'+temp_folder_interimage+'" "" Baatz "'+str(euc_threshold)+'" "@area_min@" "'+str(compactness)+'" "'+str(baatz_color)+'" "'+str(scale)+'" "'+str(bands_str)+ '" "' + str(weights_str)+'" "'+output_file+'" "seg" "0.2" "" "" "no"')

    #removing the raw file if existed
    if os.path.exists(output_file + '.raw'):
        os.remove(output_file +'.raw')
    
    #changing plm to raw
    os.rename(output_file + '.plm', output_file + ".raw")

    #removing the header lines from the raw file
    with open(output_file + ".raw", 'r+b') as f:
        lines = f.readlines()
    f.close()    
    #print len(lines)
    
    lines[:] = lines[4:]
    with open(output_file + ".raw", 'w+b') as f:
        f.write(''.join(lines))
    f.close()

    ##memory mapping
    segments_baatz = np.memmap(output_file + ".raw", dtype=np.int32, shape=(rows, cols))#uint8, float64, int32, int16, int64
    
    return segments_baatz


def region_growing_interimage(input_raster,euc_threshold,compactness,baatz_color,scale,index):#,input_bands,input_weights,output folder,reliability)
    
    '''Region growing segmentation from InterImage/TerraAIDA library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param temp_folder: path of the folder that will contain temporary files (string)
    :param exe_folder: path of the folder containing the exe files (string)
    :param euc_threshold: Euclidean distance threshold. The minimum Euclidean Distance between each segment feature. (float, positive)
    :param compactness: Baatz Compactness Weight attribute (float, between 0 and 1)
    :param baatz_color: Baatz Color Weight attribute (float, between 0 and 1)
    :param scale: Baatz scale attribute (float, positive)
    :param index: progressive index in case of multiple segmentations to differentiate temp files (integer)
    :returns:  2darray with the result of the segmentation (numpy array)
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://interimage.sourceforge.net/
               http://www.dpi.inpe.br/terraaida/
               http://marte.sid.inpe.br/col/sid.inpe.br/deise/1999/02.05.09.30/doc/T205.pdf
    '''
    
    #TODO: Would exclude this function from the package as it is not fully open-source and needs to call an exe file.
    
    #default values, set in case of 0 as input
    if euc_threshold == 0:
        euc_threshold = 50
    if compactness  == 0:
        compactness = 0.5
    if baatz_color  == 0:
        baatz_color = 0.5
    if scale  == 0:
        scale = 80
    # open the input file 
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    
    if nbands > 1:
        bands_str = ','.join(str(i) for i in xrange(nbands))
        weights_str = ','.join('1' for i in xrange(nbands))
    else:
        bands_str = '0'
        weights_str = '1'
    
    GW=geo_transform[0]
    GN=geo_transform[3]
    a= pixel2world(geo_transform, cols,rows)
    #print a[0], a[1]
    GE= a[0]
    GS= a[1]
    output_file = temp_folder_interimage + separator +'regiongrowing_' + str(euc_threshold) + '_' + str(compactness) + '_' + str(baatz_color) + '_' + str(scale)
    
    #removing the changing name file created by the segmenter after each run
    Folder_files = os.listdir(temp_folder_interimage)
    file_ = [s for s in Folder_files if "ta_segmenter" in s or "baatz" in s or "regiongrowing" in s]
    for f in file_:
        os.remove(temp_folder_interimage+separator+f)
        
    #exe_file =  exe_folder_interimage +separator+ 'ta_regiongrowing_segmenter.exe'   
    exe_file = (exe_folder_interimage +separator+ 'ta_regiongrowing_segmenter.op' if os.name == "posix" else exe_folder_interimage +separator+ 'ta_regiongrowing_segmenter.exe')
    
    #runs the regiongrowing segmenter
    os.system(exe_file + ' "'+input_raster+'" "'+str(GW)+'" "'+str(GN)+'" "'+str(GE)+'" "'+str(GS)+'" "" "'+temp_folder_interimage+'" "" RegionGrowing "'+str(euc_threshold)+'" "@area_min@" "'+str(compactness)+'" "'+str(baatz_color)+'" "'+str(scale)+'" "'+str(bands_str)+ '" "' + str(weights_str)+'" "'+output_file+'" "seg" "0.2" "" "" "no"')

    #removing the raw file if existed
    if os.path.isfile(output_file + '.raw'):
        os.remove(output_file +'.raw')

    #changing plm to raw
    os.rename(output_file + '.plm', output_file + ".raw")
        
    #removing the header lines from the raw file
    with open(output_file + ".raw", 'r+b') as f:
        lines = f.readlines()
    f.close()
    
    lines[:] = lines[4:]
    with open(output_file + ".raw", 'w+b') as f:
        f.write(''.join(lines))
    #print len(lines)
    f.close()
    
    #memory mapping
    segments_regiongrowing = np.memmap(output_file + ".raw", dtype=np.int32, shape=(rows, cols))

    return segments_regiongrowing


def watershed_otb(input_raster,output_mode,output_file,threshold,level):    
    
    '''Watershed segmentation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_mode: 'raster' or 'vector' (string)
    :param output_file: path and name of the output raster or shape depending on the output mode (*.TIF,*.tiff,*.shp) (string)
    :param threshold: threshold parameter (float, 0 for default)
    :param level: level parameter (float, 0 for default)
    :returns:  an output file is created according to the specified output mode
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu113.html#x144-9030004.9.6
    ''' 

    #TODO: Would rather use again ndarrays as IO data instead of files. We can always, if needed, than use the conversion functions to produce output files. This accounts for all the OTB functions where files are used as IO. 
    
    Segmentation = otbApplication.Registry.CreateApplication("Segmentation")
    Segmentation.SetParameterString("in", input_raster)
    Segmentation.SetParameterString("mode",output_mode)
    Segmentation.SetParameterString("mode."+output_mode+".out", output_file)
    Segmentation.SetParameterString("filter","watershed")
    if (threshold!=0):
        Segmentation.SetParameterFloat("filter.watershed.threshold",threshold)
    if (level!=0):
        Segmentation.SetParameterFloat("filter.watershed.level",level)
        
    Segmentation.ExecuteAndWriteOutput()
    Segmentation = None
    
    
def meanshift_otb(input_raster,output_mode,output_file,spatial_radius,range_radius,threshold,max_iter,min_size):
    
    '''Meanshift segmentation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_mode: 'raster' or 'vector' (string)
    :param output_file: path and name of the output raster or shape depending on the output mode (*.TIF,*.tiff,*.shp) (string)
    :param spatial_radius: spatial radius parameter (integer, 0 for default)
    :param range_radius: range radius parameter (float, 0 for default)
    :param threshold: threshold parameter (float, 0 for default)
    :param max_iter: limit on number of iterations (integer, 0 for default)
    :param min_size: minimum size parameter (integer, 0 for default)
    :returns:  an output file is created according to the specified output mode
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu113.html#x144-9030004.9.6
    ''' 
    
    Segmentation = otbApplication.Registry.CreateApplication("Segmentation")
    Segmentation.SetParameterString("in", input_raster)
    Segmentation.SetParameterString("mode",output_mode)
    Segmentation.SetParameterString("mode."+output_mode+".out", output_file)
    Segmentation.SetParameterString("filter","meanshift")
    if (spatial_radius!=0):
        Segmentation.SetParameterInt("filter.meanshift.spatialr",spatial_radius)
    if (range_radius!=0):
        Segmentation.SetParameterFloat("filter.meanshift.ranger",range_radius)
    if (threshold!=0):
        Segmentation.SetParameterFloat("filter.meanshift.thres",threshold)
    if (max_iter!=0):
        Segmentation.SetParameterInt("filter.meanshift.maxiter",max_iter)
    if (min_size!=0):
        Segmentation.SetParameterInt("filter.meanshift.minsize",min_size)

    Segmentation.ExecuteAndWriteOutput()
    
    
def edison_otb(input_raster,output_mode,output_file,spatial_radius,range_radius,min_size,scale):
    
    '''Edison-Meanshift segmentation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_mode: 'raster' or 'vector' (string)
    :param output_file: path and name of the output raster or shape depending on the output mode (*.TIF,*.tiff,*.shp) (string)
    :param spatial_radius: spatial radius parameter (integer, 0 for default)
    :param range_radius: range radius parameter (float, 0 for default)
    :param min_size: minimum size parameter (integer, 0 for default)
    :param scale: scale factor (float, 0 for default)
    :returns:  an output file is created according to the specified output mode
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu113.html#x144-9030004.9.6
    ''' 
     
    Segmentation = otbApplication.Registry.CreateApplication("Segmentation")
    Segmentation.SetParameterString("in", input_raster)
    Segmentation.SetParameterString("mode",output_mode)
    Segmentation.SetParameterString("mode."+output_mode+".out", output_file)
    Segmentation.SetParameterString("filter","edison")
    if (spatial_radius!=0):
        Segmentation.SetParameterInt("filter.edison.spatialr",spatial_radius)
    if (range_radius!=0):
        Segmentation.SetParameterFloat("filter.edison.ranger",range_radius)
    if (min_size!=0):
        Segmentation.SetParameterInt("filter.edison.minsize",min_size)
    if (scale!=0):
        Segmentation.SetParameterFloat("filter.edison.scale",scale)

    Segmentation.ExecuteAndWriteOutput()
    Segmentation = None
    
def mprofiles_otb(input_raster,output_mode,output_file,size,start,step,sigma):
    
    '''Morphological profiles segmentation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_mode: 'raster' or 'vector' (string)
    :param output_file: path and name of the output raster or shape depending on the output mode (*.TIF,*.tiff,*.shp) (string)
    :param size: profile size (integer, 0 for default)
    :param start: initial radius (integer, 0 for default)
    :param step: radius step (integer, 0 for default)
    :param sigma: threshold of the final decision rule (float, 0 for default)
    :returns:  an output file is created according to the specified output mode
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu113.html#x144-9030004.9.6
    '''
    
    Segmentation = otbApplication.Registry.CreateApplication("Segmentation")
    Segmentation.SetParameterString("in", input_raster)
    Segmentation.SetParameterString("mode",output_mode)
    Segmentation.SetParameterString("mode."+output_mode+".out", output_file)
    Segmentation.SetParameterString("filter","mprofiles")
    if (size!=0):
        Segmentation.SetParameterInt("filter.mprofiles.size",size)
    if (start!=0):
        Segmentation.SetParameterInt("filter.mprofiles.start",start)
    if (step!=0):
        Segmentation.SetParameterInt("filter.mprofiles.step",step)
    if (sigma!=0):
        Segmentation.SetParameterFloat("filter.mprofiles.sigma",sigma)

    Segmentation.ExecuteAndWriteOutput()
