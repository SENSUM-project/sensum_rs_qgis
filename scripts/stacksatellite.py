#!/usr/bin/python
import config
import os,sys
import shutil
import time
import tempfile
import osgeo.gdal, gdal
import osgeo.ogr, ogr
from osgeo.gdalconst import *
import numpy as np
import math
import argparse
import warnings
from utils import Bar

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
from sensum_library.preprocess import *
from sensum_library.classification import *
from sensum_library.segmentation import *
from sensum_library.conversion import *
from sensum_library.segmentation_opt import *
from sensum_library.features import *
from sensum_library.secondary_indicators import *
from sensum_library.multi import *

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    sat_folder = str(arg.sat_folder)
    segmentation_name = str(arg.segmentation_name)
    n_classes = int(arg.n_classes)
    coregistration = bool(arg.coregistration)
    builtup_index_method = bool(arg.builtup_index_method)
    pca_index_method = bool(arg.pca_index_method)
    pca_classification_method = bool(arg.pca_classification_method)
    dissimilarity_method = bool(arg.dissimilarity_method)
    pca_ob_method = bool(arg.pca_ob_method)
    if arg.restrict_to_city:
        restrict_to_city = True
        input_shapefile = arg.restrict_to_city[0]
    else:
        restrict_to_city = False
        input_shapefile = None
    ref_dir = (arg.ref_dir[0] if arg.ref_dir else None)
    segmentation_paramaters = map(float, arg.segmentation_paramaters)
    print segmentation_paramaters
    stacksatellite(
        sat_folder,
        input_shapefile,
        segmentation_name,
        n_classes,
        ref_dir,
        restrict_to_city,
        coregistration,
        builtup_index_method,
        pca_index_method,
        pca_classification_method,
        dissimilarity_method,
        pca_ob_method,
        segmentation_paramaters=segmentation_paramaters)

def args():
    parser = argparse.ArgumentParser(description='Stack Satellite')
    parser.add_argument("sat_folder", help="????")
    parser.add_argument("segmentation_name", help="????")
    parser.add_argument("n_classes", help="????")
    parser.add_argument("--coregistration", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--builtup_index_method", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--pca_index_method", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--pca_classification_method", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--dissimilarity_method", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--pca_ob_method", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--ref_dir", nargs=1, help="????")
    parser.add_argument("--restrict_to_city", nargs=1, help="????")
    parser.add_argument("--segmentation_paramaters", nargs="+", help="????")
    args = parser.parse_args()
    return args

def stacksatellite(sat_folder,
    input_shapefile,
    segmentation_name,
    n_classes,
    ref_dir,
    restrict_to_city,
    coregistration,
    builtup_index_method,
    pca_index_method,
    pca_classification_method,
    dissimilarity_method,
    pca_ob_method,
    segmentation_paramaters=None):
    #print segmentation_paramaters
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'

    sat_folder = sat_folder + separator     
    data_type = np.uint16
    start_time=time.time()
    dirs = os.listdir(sat_folder) #list of folders inside the satellite folder

    band_list = []
    cd_names = []
    built_up_area_pca_list = []
    built_up_area_list = []
    dissimilarity_list = []

    #reference image - if not defined the first in alphabetic order is chosen
    c = 0
    dirs.sort()
    dirs = [dir for dir in dirs if os.path.isdir(sat_folder+dir)]
    if ref_dir is None or ref_dir == '': ref_dir = sat_folder+dirs[-1]+separator
    print ref_dir
    ref_files = os.listdir(ref_dir)
    if restrict_to_city == True: #Clip the original images with the provided shapefile
        ref_list = [s for s in ref_files if ".TIF" in s and not "_city" in s and "aux.xml" not in s] #look for original landsat files
        
        for j in range(0,len(ref_list)):
            ##print ref_list[j]
            clip_rectangular(ref_dir+ref_list[j],data_type,input_shapefile,ref_dir+ref_list[j][:-4]+'_city.TIF')
        ref_files = os.listdir(ref_dir)
        ref_list = [s for s in ref_files if "_city.TIF" in s and "aux.xml" not in s]
    else: 
        ref_list = [s for s in ref_files if ".TIF" in s and "aux.xml" not in s]
    ##print ref_list

    for n in range(0,len(ref_list)):
        band_ref = read_image(ref_dir+ref_list[n],data_type,0)
        band_list.append(band_ref[0])
       
    rows_ref,cols_ref,nbands_ref,geo_transform_ref,projection_ref = read_image_parameters(ref_dir+ref_list[0])
    
    ##print len(band_list)
    if len(band_list) < 10:
        ##print 'not re-adjusted to match L8'
        band_list = normalize_to_L8(band_list)
    elif len(band_list) > 10:
        #band_list[0],band_list[1],band_list[2],band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[8]= band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9]
        #band_list[0],band_list[1],band_list[2],band_list[3],band_list[4],band_list[5]= band_list[1],band_list[2],band_list[3],band_list[4],band_list[5],band_list[9]
        new_list = (band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9])
        band_list = new_list
        
    #pca needs bands 1,2,3,4,5 or bands 1,2,3,4,5,7
    #indexes need bands 1,2,3,4,5,7
        
    if builtup_index_method == True or dissimilarity_method == True or pca_ob_method == True:
        features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
        features_list[3] = features_list[3]*1000
        features_list[4] = features_list[4]*1000
        mask_water = np.less(features_list[3],features_list[2]) #exclude water
        features_list[4] = np.choose(mask_water,(features_list[4],0))
        write_image(features_list,np.float32,0,ref_dir+'built_up_index.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
        if builtup_index_method == True:
            mask_vegetation = np.greater(features_list[2],features_list[0]) #exclude vegetation
            mask_soil = np.greater(features_list[3]/1000,0) #exclude soil
            built_up_area = np.choose(np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation)),(features_list[4]/1000,0))
            built_up_area_list.append(built_up_area)
            
    if pca_index_method == True or pca_classification_method == True or pca_ob_method == True:
        input_pca_list = (band_list[0],band_list[1],band_list[2],band_list[3],band_list[4])
        pca_mean,pca_mode,pca_second_order,pca_third_order = pca(input_pca_list) 
        pca_built_up = pca_index(pca_mean,pca_mode,pca_second_order,pca_third_order)
        
        if pca_index_method == True:
            mask_water = np.less(pca_second_order,pca_mean) #exclude water
            mask_vegetation = np.greater(pca_third_order,pca_second_order) #exclude vegetation
            mask_soil = np.less(pca_built_up,0) #exclude soil
            built_up_area_pca = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
            built_up_area_pca_list.append(built_up_area_pca)
        
        if pca_classification_method == True or pca_ob_method == True:
            write_image((pca_mean,pca_mode,pca_second_order,pca_third_order,pca_built_up),np.float32,0,ref_dir+'pca.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref)
            unsupervised_classification_otb(ref_dir+'pca.TIF',ref_dir+'pca_unsupervised.TIF',5,10)

    write_image((band_list[0],band_list[1],band_list[2],band_list[3],band_list[6]),np.uint16,0,ref_dir+'stack.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref)        
    
    if pca_ob_method == True or dissimilarity_method == True:
        ##print 'Segmentation'
        if segmentation_name == 'Edison':
            if segmentation_paramaters == None:
                edison_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',0,0,0,0)
            else:
                edison_otb(ref_dir+'built_up_index.TIF',
                    'vector',
                    ref_dir+'built_up_index_seg.shp',
                    int(segmentation_paramaters[0]),
                    float(segmentation_paramaters[1]),
                    int(segmentation_paramaters[2]),
                    float(segmentation_paramaters[3]))
        if segmentation_name == 'Meanshift':
            if segmentation_paramaters == None:
                meanshift_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',0,0,0,0,0)  
            else:
                meanshift_otb(ref_dir+'built_up_index.TIF',
                    'vector',
                    ref_dir+'built_up_index_seg.shp',
                    int(segmentation_paramaters[0]),
                    float(segmentation_paramaters[1]),
                    float(segmentation_paramaters[2]),
                    int(segmentation_paramaters[3]),
                    int(segmentation_paramaters[4]))
    
    '''
    #Extract mode from segments
    if supervised_method == True or unsupervised_method == True:
        #built-up -> polygon around vegetation or water -> optimizer -> edison -> feature extraction mode -> unsupervised classification (4 classes)
        #Input can change according to the performance: built-up index, single band, rgb combination, panchromatic band
        class_to_segments(ref_dir+'built_up_index.TIF',ref_dir+'built_up_index_seg.shp',ref_dir+'mode.shp',4)
        class_to_segments(ref_dir+ref_list[2],ref_dir+'built_up_index_seg.shp',ref_dir+'mode_b1.shp',0)
        shp2rast(ref_dir+'mode.shp',ref_dir+'mode.TIF',rows_ref,cols_ref,'Class',0,0,0,0,0,0) #conversion of the segmentation results from shape to raster for further processing
        shp2rast(ref_dir+'mode_b1.shp',ref_dir+'mode_b1.TIF',rows_ref,cols_ref,'Class',0,0,0,0,0,0)
        unsupervised_classification_otb(ref_dir+'mode.TIF',ref_dir+'mode_class.TIF',n_classes,10)
        unsupervised_classification_otb(ref_dir+'mode_b1.TIF',ref_dir+'mode_b1_class.TIF',n_classes,10)
    '''
    if dissimilarity_method == True:
        #include Daniel's function with multiprocessing
        output_list = []
        if len(band_list) < 9:
            band_diss = (band_list[0],band_list[1],band_list[6])
        else:
            band_diss = (band_list[0],band_list[1],band_list[7])
        #band_diss = (band_list[0],band_list[1],band_list[2])
        multiproc = Multi()
        window_dimension = 7
        index = 'dissimilarity'
        quantization_factor = 64
        band_list_q = linear_quantization(band_diss,quantization_factor)
        rows_w,cols_w = band_list_q[0].shape
        ##print rows_w,cols_w
        for i in range(0,rows_w):
            multiproc.put(Task_moving(i, rows_w, cols_w, band_diss,band_list_q,window_dimension,index,quantization_factor))
        multiproc.kill()
        #Write results
        output_ft_1 = np.zeros((len(band_diss),rows_w,cols_w)).astype(np.float32)
        while rows_w:
            res = multiproc.result()
            if res.size != 1:
                res = res.reshape(res.size/4,4)
                for i in range(res.size/4):
                    tmp = res[i]
                    b,index_row,index_col,feat1 = int(tmp[0]),int(tmp[1]),int(tmp[2]),tmp[3]
                    ##print b,index_row,index_col,feat1
                    output_ft_1[b][index_row][index_col]=feat1
            rows_w -= 1
        for b in range(0,len(band_diss)):
            output_list.append(output_ft_1[b][:][:])
        ##print len(output_list)
        write_image(output_list,np.float32,0,ref_dir+'dissimilarity.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
        value_to_segments(ref_dir+'dissimilarity.TIF',ref_dir+'built_up_index_seg.shp',ref_dir+'dissimilarity.shp',operation = 'Mean')

        for b in range(0,len(output_list)):
            shp2rast(ref_dir+'dissimilarity.shp',ref_dir+'dissimilarity_mean'+str(b+1)+'.tif',rows_ref,cols_ref,'Mean'+str(b+1),pixel_width=0,pixel_height=0,x_min=0,x_max=0,y_min=0,y_max=0)
            mat = read_image(ref_dir+'dissimilarity_mean'+str(b+1)+'.tif',np.uint16,0)
            dissimilarity_list.append(mat[0])
            os.remove(ref_dir+'dissimilarity_mean'+str(b+1)+'.tif')
        write_image(dissimilarity_list,np.float32,0,ref_dir + 'dissimilarity_mean.tif',rows_ref,cols_ref,geo_transform_ref,projection_ref)
        dissimilarity_list = []
        unsupervised_classification_otb(ref_dir+'dissimilarity_mean.tif',ref_dir+'dissimilarity_mean_class.tif',n_classes,10)
        rast2shp(ref_dir+'dissimilarity_mean_class.tif',ref_dir+'dissimilarity_mean_class.shp')
        value_to_segments(ref_dir+'dissimilarity_mean_class.tif',ref_dir+'dissimilarity.shp',ref_dir+'dissimilarity_class.shp',operation='Mean')
        del output_list
        del output_ft_1
        del multiproc
        
    if pca_ob_method == True:
        pca_seg_list = []
        value_to_segments(ref_dir+'pca.TIF',ref_dir+'built_up_index_seg.shp',ref_dir+'pca.shp',operation='Mean')
        for bpca in range(0,5):
            shp2rast(ref_dir+'pca.shp',ref_dir+'pca_mean'+str(bpca+1)+'.tif',rows_ref,cols_ref,'Mean'+str(bpca+1),pixel_width=0,pixel_height=0,x_min=0,x_max=0,y_min=0,y_max=0)
            mat_pca = read_image(ref_dir+'pca_mean'+str(bpca+1)+'.tif',np.uint16,0)
            pca_seg_list.append(mat_pca[0])
            os.remove(ref_dir+'pca_mean'+str(bpca+1)+'.tif')
        write_image(pca_seg_list,np.float32,0,ref_dir + 'pca_mean.tif',rows_ref,cols_ref,geo_transform_ref,projection_ref)
        pca_seg_list = []
        unsupervised_classification_otb(ref_dir+'pca_mean.tif',ref_dir+'pca_mean_class.tif',n_classes,10)
        rast2shp(ref_dir+'pca_mean_class.tif',ref_dir+'pca_mean_class.shp')
        value_to_segments(ref_dir+'pca_mean_class.tif',ref_dir+'pca.shp',ref_dir+'pca_class.shp',operation='Mean')
        
       
    #### Target directories ####
    target_directories = list(sat_folder+directory+separator for directory in reversed(dirs) if not os.path.isfile(sat_folder+directory) and (ref_dir!=sat_folder+directory+separator))
    #print target_directories
    #target_directories = [sat_folder + '2009' + separator]
    #print target_directories
    status = Bar(len(target_directories)+1, "Processing")
    status(1)
    for target_index,target_dir in enumerate(target_directories):
        print target_dir
        band_list = []
        img_files = os.listdir(target_dir)
        if restrict_to_city == True: #Clip the original images with the provided shapefile
            target_list = [s for s in img_files if ".TIF" in s and not "_city" in s] #look for original landsat files
            
            for j in range(0,len(target_list)):
                #print target_list[j]
                #clip_rectangular(input_raster,data_type,input_shape,output_raster)
                clip_rectangular(target_dir+target_list[j],data_type,input_shapefile,target_dir+target_list[j][:-4]+'_city.TIF')
                #os.system('C:/OSGeo4W64/bin/gdalwarp.exe -q -cutline "'+ input_shapefile +'" -crop_to_cutline -of GTiff "'+target_dir+target_list[j]+'" "'+target_dir+target_list[j][:-4]+'_city.TIF"')
            target_files = os.listdir(target_dir)
            target_list = [s for s in target_files if "_city.TIF" in s and "aux.xml" not in s]
        else:
            target_files = os.listdir(target_dir) 
            target_list = [s for s in target_files if ".TIF" in s and "aux.xml" not in s]
        #print target_list
        rows,cols,nbands,geo_transform,projection = read_image_parameters(target_dir+target_list[0])
        if coregistration == True: #OpenCV needs byte values (from 0 to 255)
            F_B(sat_folder,target_dir,ref_dir)
        for n in range(0,len(target_list)):
            band_target = read_image(target_dir+target_list[n],data_type,0)
            band_list.append(band_target[0])
            
        if len(band_list) < 10:
            ##print 'not re-adjusted to match L8'
            band_list = normalize_to_L8(band_list)
        elif len(band_list) > 10:
            #table of correspondence between landsat 5/7 and 8
            '''
            L5/7  ->   L8
            band1    band2
            band2    band3
            band3    band4
            band4    band5
            band5    band6
            band7    band7
            band6_1  band10
            band6_2  band11
            
            alphabetic order is putting 10 and 11 before 1, 2, ...
            '''
            #band_list[0],band_list[1],band_list[2],band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[8]= band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9]
            #band_list[0],band_list[1],band_list[2],band_list[3],band_list[4],band_list[5]= band_list[1],band_list[2],band_list[3],band_list[4],band_list[5],band_list[9] #built-up works, mask water not
            new_list = (band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9])
            band_list = new_list

        rows_target,cols_target,nbands_target,geo_transform_target,projection_target = read_image_parameters(target_dir+target_list[0])
        #pca needs bands 1,2,3,4,5 or bands 1,2,3,4,5,7
        #indexes need bands 1,2,3,4,5,7

        if builtup_index_method == True:
            features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
            features_list[3] = features_list[3]*1000
            features_list[4] = features_list[4]*1000
            write_image(features_list,np.float32,0,target_dir+'built_up_index.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
            mask_vegetation = np.greater(features_list[2],features_list[0]) #exclude vegetation
            mask_water = np.less(features_list[3],features_list[2]) #exclude water
            mask_soil = np.greater(features_list[3]/1000,0) #exclude soil
            if builtup_index_method == True:
                built_up_area = np.choose(np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation)),(features_list[4]/1000,0))
                built_up_area_list.append(built_up_area) 
        
        if pca_index_method == True or pca_classification_method == True or pca_ob_method == True:
            input_pca_list = (band_list[0],band_list[1],band_list[2],band_list[3],band_list[4])
            pca_mean,pca_mode,pca_second_order,pca_third_order = pca(input_pca_list)   
            pca_built_up = pca_index(pca_mean,pca_mode,pca_second_order,pca_third_order)
            
            if pca_index_method == True:
                mask_water = np.less(pca_second_order,pca_mean) #exclude water
                mask_vegetation = np.greater(pca_third_order,pca_second_order) #exclude vegetation
                mask_soil = np.less(pca_built_up,0) #exclude soil
                built_up_area_pca = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
                built_up_area_pca_list.append(built_up_area_pca)
            
            if pca_classification_method == True or pca_ob_method == True:
                write_image((pca_mean,pca_mode,pca_second_order,pca_third_order,pca_built_up),np.float32,0,target_dir+'pca.TIF',rows_target,cols_target,geo_transform_target,projection_target)
                unsupervised_classification_otb(target_dir+'pca.TIF',target_dir+'pca_unsupervised.TIF',5,10)
        if len(band_list) <9:
            write_image((band_list[0],band_list[1],band_list[2],band_list[3],band_list[6]),np.uint16,0,target_dir+'stack.TIF',rows_target,cols_target,geo_transform_target,projection_target)
        else:
            write_image((band_list[0],band_list[1],band_list[2],band_list[3],band_list[7]),np.uint16,0,target_dir+'stack.TIF',rows_target,cols_target,geo_transform_target,projection_target)    

        if dissimilarity_method == True:
            #include Daniel's function with multiprocessing
            output_list = []
            if len(band_list) < 9:
                band_diss = (band_list[0],band_list[1],band_list[6])
            else:
                band_diss = (band_list[0],band_list[1],band_list[7])
            #band_diss = (band_list[0],band_list[1],band_list[2])
            multiproc = Multi()
            window_dimension = 7
            index = 'dissimilarity'
            quantization_factor = 64
            band_list_q = linear_quantization(band_diss,quantization_factor)
            rows_w,cols_w = band_list_q[0].shape
            #print rows_w,cols_w
            for i in range(0,rows_w):
                multiproc.put(Task_moving(i, rows_w, cols_w, band_diss,band_list_q,window_dimension,index,quantization_factor))
            multiproc.kill()
            #Write results
            output_ft_1 = np.zeros((len(band_diss),rows_w,cols_w)).astype(np.float32)
            while rows_w:
                res = multiproc.result()
                if res.size != 1:
                    res = res.reshape(res.size/4,4)
                    for i in range(res.size/4):
                        tmp = res[i]
                        b,index_row,index_col,feat1 = int(tmp[0]),int(tmp[1]),int(tmp[2]),tmp[3]
                        ##print b,index_row,index_col,feat1
                        output_ft_1[b][index_row][index_col]=feat1
                rows_w -= 1
            for b in range(0,len(band_diss)):
                output_list.append(output_ft_1[b][:][:])
            #print len(output_list)
            write_image(output_list,np.float32,0,target_dir+'dissimilarity.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
            value_to_segments(target_dir+'dissimilarity.TIF',ref_dir+'built_up_index_seg.shp',target_dir+'dissimilarity.shp',operation='Mean')
            for b in range(0,len(output_list)):
                shp2rast(target_dir+'dissimilarity.shp',target_dir+'dissimilarity_mean'+str(b+1)+'.tif',rows_ref,cols_ref,'Mean'+str(b+1),pixel_width=0,pixel_height=0,x_min=0,x_max=0,y_min=0,y_max=0)
                mat = read_image(target_dir+'dissimilarity_mean'+str(b+1)+'.tif',np.uint16,0)
                dissimilarity_list.append(mat[0])
                os.remove(target_dir+'dissimilarity_mean'+str(b+1)+'.tif')
            write_image(dissimilarity_list,np.float32,0,target_dir + 'dissimilarity_mean.tif',rows_target,cols_target,geo_transform_target,projection_target)
            dissimilarity_list = []
            unsupervised_classification_otb(target_dir+'dissimilarity_mean.tif',target_dir+'dissimilarity_mean_class.tif',n_classes,10)
            rast2shp(target_dir+'dissimilarity_mean_class.tif',target_dir+'dissimilarity_mean_class.shp')
            value_to_segments(target_dir+'dissimilarity_mean_class.tif',target_dir+'dissimilarity.shp',target_dir+'dissimilarity_class.shp',operation='Mean')
            del output_list
            del output_ft_1
            del multiproc
            
        if pca_ob_method == True:
            pca_seg_list = []
            value_to_segments(target_dir+'pca.TIF',ref_dir+'built_up_index_seg.shp',target_dir+'pca.shp',operation='Mean')
            for bpca in range(0,5):
                shp2rast(target_dir+'pca.shp',target_dir+'pca_mean'+str(bpca+1)+'.tif',rows_ref,cols_ref,'Mean'+str(bpca+1),pixel_width=0,pixel_height=0,x_min=0,x_max=0,y_min=0,y_max=0)
                mat_pca = read_image(target_dir+'pca_mean'+str(bpca+1)+'.tif',np.uint16,0)
                pca_seg_list.append(mat_pca[0])
                os.remove(target_dir+'pca_mean'+str(bpca+1)+'.tif')
            write_image(pca_seg_list,np.float32,0,target_dir + 'pca_mean.tif',rows_ref,cols_ref,geo_transform_ref,projection_ref)
            pca_seg_list = []
            unsupervised_classification_otb(target_dir+'pca_mean.tif',target_dir+'pca_mean_class.tif',n_classes,10)
            rast2shp(target_dir+'pca_mean_class.tif',target_dir+'pca_mean_class.shp')
            value_to_segments(target_dir+'pca_mean_class.tif',target_dir+'pca.shp',target_dir+'pca_class.shp',operation='Mean')

        status(target_index+2)
        
    if builtup_index_method == True:
        rows_out,cols_out = built_up_area_list[0].shape
        write_image(built_up_area_list,np.float32,0,sat_folder+'evolution_built_up_index.TIF',rows_out,cols_out,geo_transform_ref,projection_ref)
    if pca_index_method == True:
        rows_out,cols_out = built_up_area_pca_list[0].shape
        write_image(built_up_area_pca_list,np.float32,0,sat_folder+'evolution_pca_index.TIF',rows_out,cols_out,geo_transform_ref,projection_ref)

    end_time=time.time()
    time_total = end_time-start_time
    ##print '-----------------------------------------------------------------------------------------'
    ##print 'Total time= ' + str(time_total)
    ##print '-----------------------------------------------------------------------------------------'
    

if __name__ == "__main__":
    main()
