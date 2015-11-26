'''
.. module:: segmentation_opt
   :platform: Unix, Windows
   :synopsis: This module includes functions related to the high-level classification of multi-spectral satellite images.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia
'''
'''
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
import sys
import osgeo.gdal
from osgeo.gdalconst import *
import numpy as np
try:
    import otbApplication
except:
    raise ValueError('Missing or corrupted OrfeoToolbox package')
from skimage.segmentation import felzenszwalb, slic, quickshift
from scipy import optimize
import random
import shutil
import glob
import collections
from operator import itemgetter
from conversion import *
from segmentation import *

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'
temp_folder_interimage = 'F:\Sensum_xp\Izmir\Applications\\tmpfolder'
exe_folder_interimage = 'F:\Sensum_xp\Izmir\Applications\seg_exec'


def create_extended_patch(input_band_list,reference_band,input_band_geo_transform,ref_geo_transform,ext_factor,enable_filtering):
    
    '''Creation of an extended patch around the reference polygon to reduce the processing time
    
    :param input_band_list: list of 2darrays corresponding to bands (band 1: blue) (list of numpy arrays)
    :param reference_band: 2darray related to the reference polygon (numpy array)
    :param input_band_geo_transform: geomatrix related to the original image (geomatrix)
    :param ref_geo_transform: geomatrix related to the reference band (geomatrix)
    :param ext_factor: extension factor (e.g. 0.3 for 30%) (float) 
    :param enable_filtering: dimension filter, useful for building extraction (boolean)
    :returns:  an output matrix is created with the extended patch
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''
    
    #TODO: Please avoid os.makedirs and similar file system commands whenever possible! Again, we should work with arrays and not with files.
    #TODO: Not clear to me what this function does from looking at the description. 
    #TODO: What do you mean with extended raster?, Does it mask the raster to the reference objects?
    output_band_list = []
    
    rows,cols = input_band_list[0].shape
    rows_ref,cols_ref = reference_band.shape
    
    #Regularity factor
    r_over_c = round(float(rows_ref)/float(cols_ref))
    c_over_r = round(float(cols_ref)/float(rows_ref))
    
    #Occurence of ones for filtering
    z1=reference_band.flatten()
    x1=collections.Counter(z1)
    y1=(x1.most_common())
    y2 = sorted(y1,key=itemgetter(0))
    if len(y2)>1:
        n_zeros = float(y2[0][1])
        n_ones = float(y2[1][1])
        distr = n_ones / n_zeros
    else:
        distr = 1
        
    #Filter on regularity and occurrence of ones (used for building extraction)
    if (enable_filtering == True and r_over_c < 7 and c_over_r < 7 and distr > 0.3) or (enable_filtering == False):
        loc = world2pixel(input_band_geo_transform, ref_geo_transform[0],ref_geo_transform[3])

        #calculating the dimension of the extended raster reference object  
        starting_row =int(loc[1]-rows_ref*ext_factor)
        if starting_row < 0:
            starting_row = 0
        ending_row= int(loc[1]+(ext_factor+1)*rows_ref)
        if ending_row>rows:
            ending_row = rows-1

        starting_col=int(loc[0]-cols_ref*ext_factor)
        if starting_col < 0:
            starting_col = 0
        ending_col=int(loc[0]+(ext_factor+1)*cols_ref)
        if ending_col>cols:
            ending_col = cols-1
        
        #moving from pixel to coordinates to get the geomatrix of the extended raster
        new_origins =  pixel2world(input_band_geo_transform, starting_col, starting_row)
        
        for i in range(0,len(input_band_list)):
            output_band_list.append(input_band_list[i][starting_row:ending_row, starting_col:ending_col])
        
        patch_geo_transform = (new_origins[0],input_band_geo_transform[1],input_band_geo_transform[2],new_origins[1],input_band_geo_transform[4],input_band_geo_transform[5])  
        
        return output_band_list,patch_geo_transform
    
    else:
        return 0,0
        
        
def evaluation(input_band,reference_band,input_band_geo_transform,ref_geo_transform,select_criteria):#,ref_obj_num,band_number):#d for looping on the objects#z for looping on the bands
    
    '''Creation of an extended patch around the reference polygon to reduce the processing time
    
    :param input_band: 2darray corresponding to the segmented file (numpy array)
    :param reference_band: 2darray related to the reference polygon (numpy array)
    :param input_band_geo_transform: geomatrix related to the input band (geomatrix)
    :param select_criteria: combination used to evaluate the segmentation (integer)
    :returns: number related to the segmentation evaluation is returned (float)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''
    
    #TODO: Each function should take as input a matrix. If a file is used as input, we should use before the according conversion function. This should be done outside the function in the main file.
    #TODO: Please add references!
    #TODO: define options for select_criteria argument (you have them in comments within the code, but we need to have clear in the description)
    #TODO: Taking the sum of all criteria is correct?
    #TODO: We should probably add another function or extend this one in order to allow for not only a supervised evaluation with reference object (as is done here), but also to allow for an unsupervised one (e.g. for Landsat data)
    
    if select_criteria == 0:
        select_criteria = 4

    buildings_patches = []
          
    Patch_H,Patch_W = reference_band.shape 
                
    # get the projection from the pixel position
    loc=world2pixel(input_band_geo_transform, ref_geo_transform[0],ref_geo_transform[3])
                     
    #cutting a patch from the segmentation image appropriate to the size of the building raster
    c=input_band[loc[1]:loc[1] + Patch_H, loc[0]:loc[0] + Patch_W]
    
    #getting the segments only on the building footprint and showing it
    f=c*reference_band
    #attach the segmented footprints to a list called buildings_patches_f
    buildings_patches.append(f)
            
    # getting the dictionary for the occurrence of the different segments
    z1=input_band.flatten()
    x1=collections.Counter(z1)
    p1=[elt1 for elt1,count1 in x1.most_common(1)]
    y1=x1.most_common()
               
    for i, patch in enumerate(buildings_patches):   # i is the index and patch is the value(matrix)
        
        k=np.unique(patch)
        
        #flatten the patch to carry the area calculation
        z=patch.flatten()
        x=collections.Counter(z)
        y=x.most_common()
                
        #creating a new tuple excluding the area out of the  building footprint 
        patch_counts= []
                
        for n in range(len(y)):
            if y[n][0] > 0:
                patch_counts.append((y[n][0], y[n][1]))
                                        
        if len(patch_counts)==0:
            return 100   
                               
        #calculating the area percentage of the different segments with respect to the building footprint               
        p=[count for elt,count in patch_counts]
                
        evaluation_criteria_New=[]    
        for i in range(len(p)):
            if i==0:
                evaluation_criteria_New.append(float(p[i])/float(sum(p)+0.000001))
                                         
        image_counts = []
        for i in range(len(y1)):
            for j in range(len(k)):
                if k[j]>0 and y1[i][0] == k[j]:
                    image_counts.append((y1[i][0], y1[i][1]))
               
        extra_pixel =[]
        lost_pixel = []
        count_extra_25=0
        count_lost_25=0
        for i in range(len(image_counts)):
               
            for j in range(len(patch_counts)):
                        
                if image_counts[i][0] == patch_counts[j][0]:
                    outer_part = float(image_counts[i][1])+0.000001 - float(patch_counts[j][1])
                    R1= outer_part/float(patch_counts[j][1])
                    R2 = float(patch_counts[j][1])/outer_part
                                                
                    #second criteria
                    if R1 < 0.05:     
                        extra_pixel.append(outer_part)
                    else:
                        extra_pixel.append(0)
                    
                    #third criteria
                    if R2 < 0.05:
                        lost_pixel.append(float(patch_counts[j][1]))
                    else:
                        lost_pixel.append(0)    
                                            
                    #fourth criteria
                    if (R1 > 0.05): 
                        count_extra_25 = count_extra_25 + 1
                            
                    #fifth criteria
                    if (R2 > 0.05):
                        count_lost_25 = count_lost_25 + 1
                                            
        evaluation_criteria_New.append(extra_pixel[0]/float(sum(p)+0.000001))  
        evaluation_criteria_New.append(lost_pixel[0]/float(sum(p)+0.000001)) 
        evaluation_criteria_New.append(count_extra_25) 
        evaluation_criteria_New.append(count_lost_25) 
              
        #Criteria 1: area percentage of the area of the  biggest sub-object after excluding the extra pixels
        #Criteria 2: percentage of the area of the lost pixels
        #Criteria 3: percentage of the area of the extra pixels
        #Criteria 4: the number of the reference objects which lost(extra) more than 25 percent of the pixels
        #Criteria 5: the number of the reference objects which gained more than 25 percent of the pixels
    
    if select_criteria == 1:   
        return 1-evaluation_criteria_New[0] 
    if select_criteria == 2:
        return (1-evaluation_criteria_New[0])+evaluation_criteria_New[1]+evaluation_criteria_New[2]
    if select_criteria == 3:
        return (1-evaluation_criteria_New[0])+evaluation_criteria_New[1]+evaluation_criteria_New[2]+evaluation_criteria_New[3]+evaluation_criteria_New[4]
    if select_criteria == 4:
        return evaluation_criteria_New[3]+evaluation_criteria_New[4]
    if select_criteria == 5:
        return evaluation_criteria_New[1]+evaluation_criteria_New[2]+evaluation_criteria_New[3]+evaluation_criteria_New[4]


def bound_generator(segmentation_name):
    
    '''Definition of boundaries, initial random parameters and epsilon value according to the selected segmentation
    
    :param segmentation_name: name of the desired segmentation ('Felzenszwalb','Edison','Watershed','Baatz','Baatz_integers','Region_growing','Region_growing_integers') (string)
    :returns: a list with bound_parameters (list), random parameters (list) and epsilon (float) (in this order) (list of lists)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''
    
    #TODO: I would need a walk through the whole segmentation optimization procedure.
    #TODO: OTB segmentations are not supported by the optimization procedure?
    #TODO: Again, would exclude Baatz and Region Growing algorithms.
    
    mybounds =[]
    
    if segmentation_name == 'Felzenszwalb':
        #felzenszwalb(Input_Image, scale, sigma, min_size)
        scale_bound = [0,10] #scale bound, float
        sigma_bound = [0,1] #sigma bound, float
        #min_size_bound = [1,2] #min_size bound, int
        mybounds = [scale_bound,sigma_bound]
        
        scale = random.uniform(scale_bound[0],scale_bound[1])
        sigma = random.uniform(sigma_bound[0],sigma_bound[1])
        #min_size = random.uniform(min_size_bound[0],min_size_bound[1])
        parameters = [scale,sigma]
        ep = 0.05
        
    if segmentation_name == 'Edison':
        #spatial_radius,range_radius,min_size,scale
        spatial_radius_bound = [0,20]
        range_radius_bound = [0,10.0]
        #min_size_bound = [0,200]
        #scale_bound = [100,500]
        mybounds = [spatial_radius_bound,range_radius_bound]
        
        spatial_radius = random.randrange(spatial_radius_bound[0],spatial_radius_bound[1],5)
        range_radius = random.uniform(range_radius_bound[0],range_radius_bound[1])
        #min_size = random.randrange(min_size_bound[0],min_size_bound[1],10)
        #scale = random.randrange(scale_bound[0],scale_bound[1],20)
        parameters = [int(spatial_radius),range_radius]
        ep = 0.6
        
    if segmentation_name == 'Meanshift':
        #spatial_radius,range_radius,min_size,scale
        spatial_radius_bound = [0,20] #0,100 integer
        range_radius_bound = [0,10.0] #0,100 integer
        #min_size_bound = [0,200]
        #scale_bound = [100,500]
        mybounds = [spatial_radius_bound,range_radius_bound]
        
        spatial_radius = random.randrange(spatial_radius_bound[0],spatial_radius_bound[1],5)
        range_radius = random.uniform(range_radius_bound[0],range_radius_bound[1])
        #min_size = random.randrange(min_size_bound[0],min_size_bound[1],10)
        #scale = random.randrange(scale_bound[0],scale_bound[1],20)
        parameters = [int(spatial_radius),range_radius]
        ep = 0.6
        
    if segmentation_name == 'Watershed':
        level_bound = [0.005,0.05]
        #threshold_bound = [0.005,0.05]
        mybounds = [level_bound]
        
        level = random.uniform(level_bound[0],level_bound[1])
        #threshold = random.uniform(threshold_bound[0],threshold_bound[1])
        parameters = [level]
        ep = 0.005
        
    if segmentation_name == 'Mprofiles':
        min_size_bound = [1,20]
        mybounds = [min_size_bound]
        
        min_size = random.randrange(min_size_bound[0],min_size_bound[1],2)
        parameters = [min_size]
        ep = 1
        
    if segmentation_name == 'Baatz':
        compactness_bound = [0.05,0.94]
        baatz_color_bound = [0.05,0.94]
        mybounds = [compactness_bound,baatz_color_bound]

        compactness = random.uniform(compactness_bound[0],compactness_bound[1])
        baatz_color = random.uniform(baatz_color_bound[0],baatz_color_bound[1])
        parameters = [compactness,baatz_color]
        ep = 0.05
        
    if segmentation_name == "Baatz_integers":
        euc_threshold_bound = [250,2000]
        scale_bound = [100,500]
        mybounds = [euc_threshold_bound,scale_bound]
        
        euc_threshold = random.randrange(euc_threshold_bound[0],euc_threshold_bound[1],100)
        scale = random.randrange(scale_bound[0],scale_bound[1],50)
        parameters = [int(euc_threshold),int(scale)]
        ep = 20
        
    if segmentation_name == 'Region_growing':
        compactness_bound = [0.05,0.94]
        baatz_color_bound = [0.05,0.94]
        mybounds = [compactness_bound,baatz_color_bound]
        
        compactness = random.uniform(compactness_bound[0],compactness_bound[1])
        baatz_color = random.uniform(baatz_color_bound[0],baatz_color_bound[1])
        parameters = [compactness,baatz_color]
        ep = 0.05
        
    if segmentation_name == "Region_growing_integers":
        euc_threshold_bound = [25000,50000]
        scale_bound = [100,500]
        mybounds = [euc_threshold_bound,scale_bound]
        
        euc_threshold = random.randrange(euc_threshold_bound[0],euc_threshold_bound[1],2500)
        scale = random.randrange(scale_bound[0],scale_bound[1],20)
        parameters = [int(euc_threshold),int(scale)]
        ep = 20
        
    return mybounds,parameters,ep
        
        
def optimizer(parameters,segmentation_name,patches_list,reference_band_list,patches_geo_transform_list,ref_geo_transform_list,projection,select_criteria):  
    
    '''Process to optimize the selected segmentation for multiple reference data
    
    :param parameters: list of parameters related to the segmentation to be optimized (list of floats or integers)
    :param segmentation_name: name of the desired segmentation ('Felzenszwalb','Edison','Watershed','Baatz','Baatz_integers','Region_growing','Region_growing_integers') (string)
    :param patches_list: list of created patches (list of lists)
    :param reference_band: 2darray related to the reference polygon (numpy array)
    :param input_band_geo_transform: geomatrix related to the input band (geomatrix)
    :param select_criteria: combination used to evaluate the segmentation (integer)
    :returns: number related to the segmentation evaluation is returned (float)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''
    
    #print segmentation_name,parameters
    path = os.getcwd() #working path used to save temp files
    sum_eval_criteria = 0
    
    
    if segmentation_name == 'Felzenszwalb':
        print 'working...'
        for i in range(0,len(patches_list)):
            seg = felzenszwalb_skimage(patches_list[i], parameters[0], parameters[1], 0)
            eval_criteria = evaluation(seg,reference_band_list[i],patches_geo_transform_list[i],ref_geo_transform_list[i],select_criteria)
            sum_eval_criteria = sum_eval_criteria + eval_criteria
    else:
        for i in range(0,len(patches_list)):
            rows,cols = patches_list[i][0].shape
            write_image(patches_list[i],np.uint16,0,path + separator + 'temp.tif',rows,cols,patches_geo_transform_list[i],projection)
            if os.path.isfile(path + separator + 'temp.tif'):
                #print 'working...'
                if segmentation_name == 'Edison':
                    edison_otb(path + separator + 'temp.tif','raster',path + separator + 'temp_seg.tif',int(round(parameters[0])),float(parameters[1]),0,0)
                    seg_list = read_image(path + separator + 'temp_seg.tif',np.uint16,0)
                    seg = seg_list[0]
                if segmentation_name == 'Meanshift':
                    meanshift_otb(path + separator + 'temp.tif','raster',path + separator + 'temp_seg.tif',int(round(parameters[0])),float(parameters[1]),0,0,0)
                    seg_list = read_image(path + separator + 'temp_seg.tif',np.uint16,0)
                    seg = seg_list[0]
                if segmentation_name == 'Watershed':
                    watershed_otb(path + separator + 'temp.tif','raster',path + separator + 'temp_seg.tif',0,float(parameters[0]))
                    seg_list = read_image(path + separator + 'temp_seg.tif',np.uint16,0)
                    seg = seg_list[0]
                if segmentation_name == 'Mprofiles':
                    mprofiles_otb(path + separator + 'temp.tif','raster',path + separator + 'temp_seg.tif',int(parameters[0]),0,0,0)
                    seg_list = read_image(path + separator + 'temp_seg.tif',np.uint16,0)
                    seg = seg_list[0]
                if segmentation_name == 'Baatz':
                    seg = baatz_interimage(path + separator + 'temp.tif',0,float(parameters[0]),float(parameters[1]),0,i)
                if segmentation_name == 'Baatz_integers':
                    seg = baatz_interimage(path + separator + 'temp.tif',int(parameters[0]),0,0,int(parameters[1]),i)
                if segmentation_name == 'Region_growing':
                    seg = region_growing_interimage(path + separator + 'temp.tif',0,float(parameters[0]),float(parameters[1]),0,i)
                if segmentation_name == 'Region_growing_integers':
                    seg = region_growing_interimage(path + separator + 'temp.tif',int(parameters[0]),0,0,int(parameters[1]),i)
            
                eval_criteria = evaluation(seg,reference_band_list[i],patches_geo_transform_list[i],ref_geo_transform_list[i],select_criteria)
                sum_eval_criteria = sum_eval_criteria + eval_criteria
            
    return sum_eval_criteria


def call_optimizer(segmentation_name,patches_list,reference_band_list,patches_geo_transform_list,ref_geo_transform_list,projection,select_criteria,nloops):
    
    '''Call the optimizer function
    
    :param segmentation_name: name of the desired segmentation ('Felzenszwalb','Edison','Watershed','Baatz','Baatz_integers','Region_growing','Region_growing_integers') (string)
    :param patches_list: list of created patches (list of lists)
    :param reference_band: 2darray related to the reference polygon (numpy array)
    :param input_band_geo_transform: geomatrix related to the input band (geomatrix)
    :param select_criteria: combination used to evaluate the segmentation (integer)
    :param nloops: number of times the optimizer is executed using different starting points (integer)
    :returns: list of optimum parameters
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''
    
    opt_list = []
    fun_values = []
    print "nloops: ".format(nloops)
    for x in range(nloops):
        mybounds,parameters,ep = bound_generator(segmentation_name)
        ind = len(parameters)
        e = optimize.fmin_l_bfgs_b(optimizer,parameters,args=(segmentation_name,patches_list,reference_band_list,patches_geo_transform_list,ref_geo_transform_list,projection,select_criteria), bounds = mybounds,approx_grad=True, factr=10.0, pgtol=1e-20, epsilon=ep, iprint=-1, maxfun=15000, maxiter=15000, disp=None, callback=None)
        for p in range (0,len(parameters)):
            opt_list.append(e[0][p])
        fun_values.append(e[1])
        
    for i in [i for i,x in enumerate(fun_values) if x == min(fun_values)]:
        i
    min_index = i
    #print min_index
    opt_parameters=opt_list[(i)*ind:(i+1)*ind]
    print 'opt_parameters',opt_parameters
    
    path = os.getcwd()
    if os.path.isfile(path+separator+'temp.tif'):
        os.remove(path+separator+'temp.tif')
    if os.path.isfile(path+separator+'temp_seg.tif'):
        os.remove(path+separator+'temp_seg.tif')
    
    return opt_parameters