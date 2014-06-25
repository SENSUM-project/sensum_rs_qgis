# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sensum
                                 A QGIS plugin
 Sensum QGIS Plugin
                              -------------------
        begin                : 2014-05-27
        copyright            : (C) 2014 by Eucentre
        email                : dgaleazzo@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
# Import the PyQt and QGIS libraries
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from qgis.core import *
# Initialize Qt resources from file resources.py
import resources_rc
# Import the code for the dialog
from sensumdialog import *
import os,sys
import os.path
import shutil
import time
import tempfile
import osgeo.gdal, gdal
import osgeo.ogr
import numpy as np
import math
import scipy.signal
import scipy as sp
import cv2
from osgeo.gdalconst import *
from sensum_library.preprocess import *
from sensum_library.classification import *
from sensum_library.segmentation import *
from sensum_library.conversion import *
from sensum_library.segmentation_opt import *
from sensum_library.features import *
from sensum_library.secondary_indicators import *
from sensum_library.multi import *
from numpy.fft import fft2, ifft2, fftshift
from skimage.morphology import square, closing
import otbApplication

def stacksatellite(sat_folder,input_shapefile,quantization_mode,opt_polygon,segmentation_name,select_criteria,nloops,n_classes,ref_dir,restrict_to_city,coregistration,builtup_index_method,pca_index_method,pca_classification_method,dissimilarity_method,band_combination,supervised_method,supervised_polygon):
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'

    sat_folder = sat_folder + separator     
    data_type = np.uint16
    start_time=time.time()
    dirs = os.listdir(sat_folder) #list of folders inside the satellite folder
    print 'List of files and folders: ' + str(dirs)

    band_ref_uint8 = []
    band_list = []
    built_up_area_pca_list = []
    built_up_area_list = []
    dissimilarity_list = []

    #reference image - if not defined the first in alphabetic order is chosen
    c = 0

    if ref_dir is None or ref_dir == '': #if a reference image is not provided, the first in alphabetic order is chosen
        print 'Reference directory not specified - The first folder in alphabetical order will be chosen'
        while (os.path.isfile(dirs[c]) == True):### to avoid taking the files in the dirs as a reference folder so, the program will search for the first folder
            c=c+1
        else:
            reference_dir = dirs[c]
        ref_dir = sat_folder + reference_dir + separator #first directory assumed to be the reference
    ref_files = os.listdir(ref_dir)

    if restrict_to_city == True: #Clip the original images with the provided shapefile
        ref_list = [s for s in ref_files if ".TIF" in s and not "_city" in s] #look for original landsat files
        
        for j in range(0,len(ref_list)):
            print ref_list[j]
            #clip_rectangular(input_raster,data_type,input_shape,output_raster)
            clip_rectangular(ref_dir+ref_list[j],np.uint8,input_shapefile,ref_dir+ref_list[j][:-4]+'_city.TIF')
        
        ref_files = os.listdir(ref_dir)
        ref_list = [s for s in ref_files if "_city.TIF" in s]
    else: 
        ref_list = [s for s in ref_files if ".TIF" in s]
    print ref_list
    for n in range(0,len(ref_list)):
        band_ref = read_image(ref_dir+ref_list[n],data_type,0)
        band_list.append(band_ref[0])
    rows_ref,cols_ref,nbands_ref,geo_transform_ref,projection_ref = read_image_parameters(ref_dir+ref_list[0])
    print len(band_list)
    #pca needs bands 1,2,3,4,5 or bands 1,2,3,4,5,7
    #indexes need bands 1,2,3,4,5,7
    if builtup_index_method == True or supervised_method == True or unsupervised_method == True:
        features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
        features_list[3] = features_list[3]*1000
        features_list[4] = features_list[4]*1000
        write_image(features_list,np.float32,0,ref_dir+'built_up_index.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file

    if builtup_index_method == True:
        mask_vegetation = np.greater(features_list[1],features_list[0]) #exclude vegetation
        mask_water = np.less(features_list[2],features_list[1]) #exclude water
        mask_soil = np.greater(features_list[3],0) #exclude soil
        built_up_area = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
        built_up_area_list.append(built_up_area) 

    if pca_index_method == True or pca_classification_method == True:
        input_pca_list = (band_list[0],band_list[1],band_list[2],band_list[3],band_list[4])
        pca_mean,pca_mode,pca_second_order,pca_third_order = pca(input_pca_list)
        
        pca_built_up = pca_index(pca_mean,pca_mode,pca_second_order,pca_third_order)
        
        if pca_index_method == True:
            mask_water = np.less(pca_second_order,pca_mean) #exclude water
            mask_vegetation = np.greater(pca_third_order,pca_second_order) #exclude vegetation
            mask_soil = np.less(pca_built_up,0) #exclude soil
            built_up_area_pca = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
            built_up_area_pca_list.append(built_up_area_pca)
        
        if pca_classification_method == True:
            write_image((pca_mean,pca_mode,pca_second_order,pca_third_order,pca_built_up),data_type,0,ref_dir+'pca.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref)
            unsupervised_classification_otb(ref_dir+'pca.TIF',ref_dir+'pca_unsupervised.TIF',5,10)
            
    if supervised_method == True:
        print 'Segmentation'
        '''
        driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
        inDS = driver_shape.Open(input_shapefile, 0)
        if inDS is None:
            print 'Could not open file'
            sys.exit(1)
        inLayer = inDS.GetLayer()
        temp = split_shape(inLayer,0)
        temp_layer = temp.GetLayer()
        reference_polygon_matrix, ref_polygon_geo_transform = polygon2array(temp_layer,geo_transform_ref[1],abs(geo_transform_ref[5])) 
        temp.Destroy() 
        
        ext_patch_list,patch_geo_transform = create_extended_patch(band_list,reference_polygon_matrix,geo_transform_ref,ref_polygon_geo_transform,0.3,False)   
        e = call_optimizer(segmentation_name,[ext_patch_list],[reference_polygon_matrix],[patch_geo_transform],[ref_polygon_geo_transform],projection_ref,select_criteria,nloops)
        '''
        if segmentation_name == 'Edison':
            #edison_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',int(e[0]),float(e[1]),0,0)
            edison_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',0,0,0,0)
        if segmentation_name == 'Meanshift':
            #meanshift_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',int(e[0]),float(e[1]),0,0,0)
            meanshift_otb(ref_dir+'built_up_index.TIF','vector',ref_dir+'built_up_index_seg.shp',0,0,0,0,0)
        #inDS.Destroy()   
             
    if unsupervised_method == True:
        #include stuff
        print 'to implement'
        
    #Extract mode from segments
    if supervised_method == True or unsupervised_method == True:
        #built-up -> polygon around vegetation or water -> optimizer -> edison -> feature extraction mode -> unsupervised classification (4 classes)
        #Input can change according to the performance: built-up index, single band, rgb combination, panchromatic band
        class_to_segments(ref_dir+'built_up_index.TIF',ref_dir+'built_up_index_seg.shp',ref_dir+'mode.shp')
        shp2rast(ref_dir+'mode.shp',ref_dir+'mode.TIF',rows_ref,cols_ref,'Class',0,0,0,0,0,0) #conversion of the segmentation results from shape to raster for further processing
        unsupervised_classification_otb(ref_dir+'mode.TIF',ref_dir+'mode_class.TIF',n_classes,1)
        
        #Define if a vegetation filter is needed
        '''
        mask_veg = np.less(NDBI-SAVI,0) 
        WriteOutputImage(ref_dir+ref_list_city[0],ref_dir,'','vegetation_mask.TIF',0,0,0,1,[SAVI])
        veg_opening = binary_opening(mask_veg,square(5))
        WriteOutputImage(ref_dir+ref_list_city[0],ref_dir,'','vegetation_mask_opening.TIF',0,0,0,1,[veg_opening])
        veg_filt = np.equal(veg_opening,0)
        out_veg_filt = np.choose(veg_filt,(0,(list_mode_class[0])))
        '''
    if dissimilarity_method == True:
        #include Daniel's function with multiprocessing
        output_list = []
        if len(band_list) < 9:
            band_diss = (band_list[0],band_list[1],band_list[6])
        else:
            band_diss = (band_list[0],band_list[1],band_list[7])
        multiproc = Multi()
        window_dimension = 7
        index = 'dissimilarity'
        quantization_factor = 64
        band_list_q = linear_quantization(band_diss,quantization_factor)
        rows_w,cols_w = band_list_q[0].shape
        print rows_w,cols_w
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
                    #print b,index_row,index_col,feat1
                    output_ft_1[b][index_row][index_col]=feat1
            rows_w -= 1
        for b in range(0,len(band_diss)):
            output_list.append(output_ft_1[b][:][:])
        print len(output_list)
        write_image(output_list,np.float32,0,ref_dir+'dissimilarity.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
        dissimilarity_list.append(output_list)
        del output_list
        del output_ft_1
        del multiproc

    for i in range(0,len(dirs)):
        band_list = []
        if (os.path.isfile(sat_folder+dirs[i]) == False) and ((ref_dir!=sat_folder+dirs[i]+separator)):
            target_dir = sat_folder+dirs[i]+separator
            img_files = os.listdir(target_dir)
            
            if restrict_to_city == True: #Clip the original images with the provided shapefile
                target_list = [s for s in img_files if ".TIF" in s and not "_city" in s] #look for original landsat files
                
                for j in range(0,len(target_list)):
                    print target_list[j]
                    #clip_rectangular(input_raster,data_type,input_shape,output_raster)
                    clip_rectangular(target_dir+target_list[j],np.uint8,input_shapefile,target_dir+target_list[j][:-4]+'_city.TIF')
                
                target_files = os.listdir(target_dir)
                target_list = [s for s in target_files if "_city.TIF" in s and "aux.xml" not in s]
            else: 
                target_list = [s for s in target_files if ".TIF" in s]
            print target_list
            rows,cols,nbands,geo_transform,projection = read_image_parameters(target_dir+target_list[0])
            if coregistration == True: #OpenCV needs byte values (from 0 to 255)
                F_B(sat_folder,target_dir,ref_dir)
            for n in range(0,len(target_list)):
                band_target = read_image(target_dir+target_list[n],data_type,0)
                band_list.append(band_target[0])
            print len(band_list)
            rows_target,cols_target,nbands_target,geo_transform_target,projection_target = read_image_parameters(target_dir+target_list[0])
            #pca needs bands 1,2,3,4,5 or bands 1,2,3,4,5,7
            #indexes need bands 1,2,3,4,5,7
            if builtup_index_method == True or supervised_method == True or unsupervised_method == True:
                features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
                features_list[3] = features_list[3]*1000
                features_list[4] = features_list[4]*1000
                write_image(features_list,np.float32,0,target_dir+'built_up_index.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
            
            if builtup_index_method == True:
                mask_vegetation = np.greater(features_list[1],features_list[0]) #exclude vegetation
                mask_water = np.less(features_list[2],features_list[1]) #exclude water
                mask_soil = np.greater(features_list[3],0) #exclude soil
                built_up_area = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
                built_up_area_list.append(built_up_area) 
            
            if pca_index_method == True or pca_classification_method == True:
                input_pca_list = (band_list[0],band_list[1],band_list[2],band_list[3],band_list[4])
                pca_mean,pca_mode,pca_second_order,pca_third_order = pca(input_pca_list)
                
                pca_built_up = pca_index(pca_mean,pca_mode,pca_second_order,pca_third_order)
                
                if pca_index_method == True:
                    mask_water = np.less(pca_second_order,pca_mean) #exclude water
                    mask_vegetation = np.greater(pca_third_order,pca_second_order) #exclude vegetation
                    mask_soil = np.less(pca_built_up,0) #exclude soil
                    built_up_area_pca = np.logical_and(mask_soil,np.logical_and(mask_water,mask_vegetation))
                    built_up_area_pca_list.append(built_up_area_pca)
                
                if pca_classification_method == True:
                    write_image((pca_mean,pca_mode,pca_second_order,pca_third_order,pca_built_up),data_type,0,target_dir+'pca.TIF',rows_target,cols_target,geo_transform_target,projection_target)
                    unsupervised_classification_otb(target_dir+'pca.TIF',target_dir+'pca_unsupervised.TIF',5,10)
                    
            if supervised_method == True:
                print 'Segmentation'
                '''
                driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
                inDS = driver_shape.Open(input_shapefile, 0)
                if inDS is None:
                    print 'Could not open file'
                    sys.exit(1)
                inLayer = inDS.GetLayer()
                temp = split_shape(inLayer,0)
                temp_layer = temp.GetLayer()
                reference_polygon_matrix, ref_polygon_geo_transform = polygon2array(temp_layer,geo_transform_target[1],abs(geo_transform_target[5])) 
                temp.Destroy() 
                
                ext_patch_list,patch_geo_transform = create_extended_patch(band_list,reference_polygon_matrix,geo_transform_target,ref_polygon_geo_transform,0.3,False)   
                e = call_optimizer(segmentation_name,[ext_patch_list],[reference_polygon_matrix],[patch_geo_transform],[ref_polygon_geo_transform],projection_target,select_criteria,nloops)
                '''
                if segmentation_name == 'Edison':
                    #(input_raster,output_mode,output_file,spatial_radius,range_radius,min_size,scale)
                    #edison_otb(target_dir+'built_up_index.TIF','vector',target_dir+'built_up_index_seg.shp',int(e[0]),float(e[1]),0,0)
                    edison_otb(target_dir+'built_up_index.TIF','vector',target_dir+'built_up_index_seg.shp',0,0,0,0)
                if segmentation_name == 'Meanshift':
                    #meanshift_otb(input_raster,output_mode,output_file,spatial_radius,range_radius,threshold,max_iter,min_size)
                    #meanshift_otb(target_dir+'built_up_index.TIF','vector',target_dir+'built_up_index_seg.shp',int(e[0]),float(e[1]),0,0,0)
                    meanshift_otb(target_dir+'built_up_index.TIF','vector',target_dir+'built_up_index_seg.shp',0,0,0,0,0)
                #inDS.Destroy()   
                     
            if unsupervised_method == True:
                #include stuff
                print 'to implement'
                
            #Extract mode from segments
            if supervised_method == True or unsupervised_method == True:
                #built-up -> polygon around vegetation or water -> optimizer -> edison -> feature extraction mode -> unsupervised classification (4 classes)
                #Input can change according to the performance: built-up index, single band, rgb combination, panchromatic band
                class_to_segments(target_dir+'built_up_index.TIF',target_dir+'built_up_index_seg.shp',target_dir+'mode.shp')
                shp2rast(target_dir+'mode.shp',target_dir+'mode.TIF',rows_target,cols_target,'Class',0,0,0,0,0,0) #conversion of the segmentation results from shape to raster for further processing
                unsupervised_classification_otb(target_dir+'mode.TIF',target_dir+'mode_class.TIF',n_classes,1)
                
                #Define if a vegetation filter is needed
                '''
                mask_veg = np.less(NDBI-SAVI,0) 
                WriteOutputImage(ref_dir+ref_list_city[0],ref_dir,'','vegetation_mask.TIF',0,0,0,1,[SAVI])
                veg_opening = binary_opening(mask_veg,square(5))
                WriteOutputImage(ref_dir+ref_list_city[0],ref_dir,'','vegetation_mask_opening.TIF',0,0,0,1,[veg_opening])
                veg_filt = np.equal(veg_opening,0)
                out_veg_filt = np.choose(veg_filt,(0,(list_mode_class[0])))
                '''
            if dissimilarity_method == True:
                #include Daniel's function with multiprocessing
                output_list = []
                if len(band_list) < 9:
                    band_diss = (band_list[0],band_list[1],band_list[6])
                else:
                    band_diss = (band_list[0],band_list[1],band_list[7])
                multiproc = Multi()
                window_dimension = 7
                index = 'dissimilarity'
                quantization_factor = 64
                band_list_q = linear_quantization(band_diss,quantization_factor)
                rows_w,cols_w = band_list_q[0].shape
                print rows_w,cols_w
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
                            #print b,index_row,index_col,feat1
                            output_ft_1[b][index_row][index_col]=feat1
                    rows_w -= 1
                for b in range(0,len(band_diss)):
                    output_list.append(output_ft_1[b][:][:])
                print len(output_list)
                write_image(output_list,np.float32,0,target_dir+'dissimilarity.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
                dissimilarity_list.append(output_list)
                del output_list
                del output_ft_1
                del multiproc
                
    if builtup_index_method == True:
        write_image(built_up_area_list,data_type,0,sat_folder+'evolution_built_up_index.TIF',rows,cols,geo_transform,projection)
    if pca_index_method == True:
        write_image(built_up_area_pca_list,data_type,0,sat_folder+'evolution_pca_index.TIF',rows,cols,geo_transform,projection)
    if dissimilarity_method == True:
        write_image(dissimilarity_list,data_type,0,sat_folder+'evolution_dissimilarity.TIF',rows,cols,geo_transform,projection)

    end_time=time.time()
    time_total = end_time-start_time
    print '-----------------------------------------------------------------------------------------'
    print 'Total time= ' + str(time_total)
    print '-----------------------------------------------------------------------------------------'


def footprints(pansharp_file,training_set,training_attribute,building_classes,output_shape,enable_smooth_filter=True,ui_progress=None):

    #pansharp_file = 'F:/Sensum_xp/Izmir/building_footprints/New_development/pansharp.tif'
    #training_set = 'F:/Sensum_xp/Izmir/building_footprints/New_development/training.shp' #supervised classification
    #training_attribute = 'Class'
    #building_classes = [7,8,9]

    if ui_progress:
        ui_progress.progressBar.setMinimum(1)
        ui_progress.progressBar.setMaximum(500)
        ui_progress.label_title.setText("(1/5) Smooth filter...this may take a while")
        ui_progress.progressBar.setValue(1)
    #Apply smooth filter to original image
    if enable_smooth_filter == True:
        print 'Smooth filter...this may take a while'
        smooth_filter_otb(pansharp_file,pansharp_file[:-4]+'_smooth.tif',30)
        process_file = pansharp_file[:-4]+'_smooth.tif'
    else:
        process_file = pansharp_file
    if ui_progress:
        ui_progress.label_title.setText("(2/5) Supervised classification...")
        ui_progress.progressBar.setValue(100)
    print 'Supervised classification...'
    train_classifier_otb([process_file],[training_set],pansharp_file[:-4]+'_svm.txt','svm',training_attribute)
    supervised_classification_otb(process_file,pansharp_file[:-4]+'_svm.txt',pansharp_file[:-4]+'_svm.tif')
    if ui_progress:
        ui_progress.label_title.setText("(3/5) Conversion to shapefile...")
        ui_progress.progressBar.setValue(200)
    print 'Conversion to shapefile...'
    rast2shp(pansharp_file[:-4]+'_svm.tif',pansharp_file[:-4]+'_svm.shp')
    if ui_progress:
        ui_progress.label_title.setText("(4/5) Area and Class filtering...")
        ui_progress.progressBar.setValue(300)
    print 'Area and Class filtering...'
    driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    driver_mem = osgeo.ogr.GetDriverByName('Memory')
    infile = driver_shape.Open(pansharp_file[:-4]+'_svm.shp') #input file
    outfile=driver_mem.CreateDataSource(pansharp_file[:-4]+'_svm_filt') #output file
    outlayer=outfile.CreateLayer('Footprint',geom_type=osgeo.ogr.wkbPolygon)
    dn_def = osgeo.ogr.FieldDefn('DN', osgeo.ogr.OFTInteger)
    area_def = osgeo.ogr.FieldDefn('Area', osgeo.ogr.OFTReal)
    outlayer.CreateField(dn_def)
    outlayer.CreateField(area_def)
    inlayer=infile.GetLayer()
    x_min, x_max, y_min, y_max = inlayer.GetExtent()
    infeature = inlayer.GetNextFeature()
    feature_def = outlayer.GetLayerDefn()
    while infeature:
        geom = infeature.GetGeometryRef()
        area = geom.Area()
        dn = infeature.GetField('DN')
        if dn in building_classes and area > 5:
            outfeature = osgeo.ogr.Feature(feature_def)
            outfeature.SetGeometry(geom)
            outfeature.SetField('DN',dn)
            outfeature.SetField('Area',area)
            outlayer.CreateFeature(outfeature)
            outfeature.Destroy()
        infeature = inlayer.GetNextFeature()
    infile.Destroy()
    if ui_progress:
        ui_progress.label_title.setText("(5/5) Conversion to shapefile...")
        ui_progress.progressBar.setV
    print 'Morphology filter...'
    rows,cols,nbands,geotransform,projection = read_image_parameters(pansharp_file)
    #Filter by area and create output shapefile
    if os.path.isfile(output_shape):
        os.remove(output_shape)
    final_file = driver_shape.CreateDataSource(output_shape) #final file
    final_layer = final_file.CreateLayer('Buildings',geom_type=osgeo.ogr.wkbPolygon)
    class_def = osgeo.ogr.FieldDefn('Class', osgeo.ogr.OFTInteger)
    area_def = osgeo.ogr.FieldDefn('Area', osgeo.ogr.OFTReal)
    final_layer.CreateField(class_def)
    final_layer.CreateField(area_def)
    feature_def_fin = final_layer.GetLayerDefn()
    for c in range(0,len(building_classes)):
        query = 'SELECT * FROM Footprint WHERE (DN = ' + str(building_classes[c]) + ')'
        filt_layer = outfile.ExecuteSQL(query)
        #Conversion to raster, forced dimensions from the original shapefile
        x_res = int((x_max - x_min) / geotransform[1]) #pixel x-axis resolution
        y_res = int((y_max - y_min) / abs(geotransform[5])) #pixel y-axis resolution
        target_ds = osgeo.gdal.GetDriverByName('MEM').Create('', x_res, y_res, GDT_Byte) #create layer in memory
        geo_transform = [x_min, geotransform[1], 0, y_max, 0, geotransform[5]] #geomatrix definition
        target_ds.SetGeoTransform(geo_transform)
        band = target_ds.GetRasterBand(1)
        # Rasterize
        osgeo.gdal.RasterizeLayer(target_ds, [1], filt_layer, burn_values=[1])
        # Read as array
        build_matrix = band.ReadAsArray()
        target_ds = None
        filt_layer = None
        #Morphology filter
        build_fill = sp.ndimage.binary_fill_holes(build_matrix, structure=None, output=None, origin=0)
        build_open = sp.ndimage.binary_opening(build_fill, structure=np.ones((3,3))).astype(np.int)
        #Conversion to shapefile
        target_ds = osgeo.gdal.GetDriverByName('MEM').Create('temp', x_res, y_res, GDT_Byte) #create layer in memory
        band = target_ds.GetRasterBand(1)
        target_ds.SetGeoTransform(geo_transform)
        band.WriteArray(build_open, 0, 0)
        build_file = driver_mem.CreateDataSource('Conversion') #output file
        build_layer=build_file.CreateLayer('Conv',geom_type=osgeo.ogr.wkbPolygon)
        dn = osgeo.ogr.FieldDefn('DN',osgeo.ogr.OFTInteger)
        build_layer.CreateField(dn)
        osgeo.gdal.Polygonize(band,band.GetMaskBand(),build_layer,0)
        #Filter by area
        build_feature = build_layer.GetNextFeature()
        nfeature = build_layer.GetFeatureCount()
        while build_feature:
            #build_feature = build_layer.GetFeature(f)
            geom = build_feature.GetGeometryRef()
            area = geom.Area()
            if area > 10 and area < 20000:
                final_feature = osgeo.ogr.Feature(feature_def_fin)
                final_feature.SetGeometry(geom)
                final_feature.SetField('Class',int(building_classes[c]))
                final_feature.SetField('Area',area)
                final_layer.CreateFeature(final_feature)
                final_feature.Destroy()
            build_feature = build_layer.GetNextFeature()
        target_ds = None
        build_layer = None
        build_file.Destroy()
    shutil.copyfile(pansharp_file[:-4]+'_svm.prj', output_shape[:-4]+'.prj')
    outfile.Destroy()
    final_file.Destroy()

def Extraction(image1,image2):
    
    '''
    ###################################################################################################################
    Feature Extraction using the SURF algorithm
    
    Input:
     - image1: path to the reference image - each following image is going to be matched with this reference
     - image2: path to the image to be corrected
    
    Output:
    Returns a matrix with x,y coordinates of matching points and the minimum distance
    ###################################################################################################################
    '''
    image_1 = osgeo.gdal.Open(image1,GA_ReadOnly)
    inband1=image_1.GetRasterBand(1)
    cols = image_1.RasterXSize
    img1 = inband1.ReadAsArray().astype('uint8')

    img1_m = np.ma.masked_values(img1, 0).astype('uint8')

    detector = cv2.FeatureDetector_create("SURF") 
    descriptor = cv2.DescriptorExtractor_create("BRIEF")
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    #print img1.type

    kp1 = detector.detect(img1, mask=img1_m)
    print img1_m
    k1, d1 = descriptor.compute(img1, kp1)

    image_1 = None
    img1 = None
    
    image_2 = osgeo.gdal.Open(image2,GA_ReadOnly)
    inband2=image_2.GetRasterBand(1)
    img2 = inband2.ReadAsArray().astype('uint8')

    img2_m = np.ma.masked_values(img2, 0).astype('uint8')
    
    kp2 = detector.detect(img2, mask = img2_m)
    k2, d2 = descriptor.compute(img2, kp2)

    image_2 = None
    img2 = None

    # match the keypoints
    matches = matcher.match(d1, d2)
    
    # visualize the matches
    dist = [m.distance for m in matches] #extract the distances
  
    thres_dist = 100
    sel_matches = [m for m in matches if m.distance <= thres_dist]

    points=np.zeros(shape=(len(sel_matches),4))
    points_shift = np.zeros(shape=(len(sel_matches),2))
    points_shift_abs = np.zeros(shape=(len(sel_matches),1))

    # Creo una variabile dove vado a scrivere, per ogni coppia: distanza hamming, shift, pendenza
   
    compar_stack = np.array([100,1.5,0.0,1,1,2,2])

    #vishualization(img1,img2,sel_matches)
    i = 0
    for m in sel_matches:
        points[i][:]= [int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1]),int(k2[m.trainIdx].pt[0]),int(k2[m.trainIdx].pt[1])]
        points_shift[i][:] = [int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]),int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])]
        points_shift_abs [i][:] = [np.sqrt((int(cols + k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))**2+
                                           (int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))**2)]
        
        #print m.distance,'   ' , [np.sqrt((int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))**2+
                                          # (int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))**2)]
        
        deltax = np.float(int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))
        deltay = np.float(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))
        
        if deltax == 0 and deltay != 0:
            slope = 90
        elif deltax == 0 and deltay == 0:
            slope = 0
        else:
            slope = (np.arctan(deltay/deltax)*360)/(2*np.pi)
        
        compar_stack = np.vstack([compar_stack,[m.distance,points_shift_abs [i][:],slope,
                                                int(k1[m.queryIdx].pt[0]),
                                                int(k1[m.queryIdx].pt[1]),
                                                int(k2[m.trainIdx].pt[0]),
                                                int(k2[m.trainIdx].pt[1])]])
        i=i+1

        
        
    #Ordino lo stack
    compar_stack = compar_stack[compar_stack[:,0].argsort()]#Returns the indices that would sort an array.
    #print compar_stack#[0:30]
    print len(compar_stack)

    best = best_row_2(compar_stack[0:90])#The number of sorted points to be passed
    print 'BEST!!!!!', best
        
    report = os.path.join(os.path.dirname(image1),'report.txt')
    out_file = open(report,"a")
    out_file.write("\n")
    out_file.write("Il migliore ha un reliability value pari a "+str(best[0])+" \n")
    out_file.close()
    migliore = [best[3:7]]
    #return migliore,best[0]
    return migliore

def best_row_2(compstack):
    '''
    ###################################################################################################################
    
    Input:
     - compstack: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    # Sort
    compstack = compstack[compstack[:,2].argsort()]
    spl_slope = np.append(np.where(np.diff(compstack[:,2])>0.1)[0]+1,len(compstack[:,0]))
    
    step = 0
    best_variability = 5
    len_bestvariab = 0
    best_row = np.array([100,1.5,0.0,1,1,2,2])

    for i in spl_slope:
        slope = compstack[step:i][:,2]
        temp = compstack[step:i][:,1]
        variab_temp = np.var(temp)
        count_list=[]
        if variab_temp <= best_variability and len(temp) >3:
            count_list.append(len(temp))

            if variab_temp < best_variability:
                
                best_variability = variab_temp
                len_bestvariab = len(temp)                
                best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                all_rows = compstack[step:i]
            if variab_temp == best_variability:
                if len(temp)>len_bestvariab:
                    best_variability = variab_temp
                    len_bestvariab = len(temp)                
                    best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                    all_rows = compstack[step:i]
        step = i
    return best_row#,,point_list1,point_list2



def affine_corrected(img,delta_x,delta_y):
    '''
    ###################################################################################################################
    
    Input:
     - img,delta_x,delta_y: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    rows,cols = img.shape  
    M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


def F_B(path,floating,ref):
    '''
    ###################################################################################################################
    
    Input:
     - path,floating,ref: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    dir1 = os.listdir(floating)
    dir2 = os.listdir(ref)
    a_list=[]
    for i in [1]:#range(1,6):#(1,6):
        ref_list = [s for s in dir1 if "_B"+str(i)+'_city' in s ]
        if len(ref_list):
            floating_list = [s for s in dir2 if "_B"+str(i)+'_city' in s ]
            rows,cols_q,nbands,geotransform,projection = read_image_parameters(floating+ref_list[0])
            rows,cols_q,nbands,geotransform,projection = read_image_parameters(ref+floating_list[0])
            band_list0 = read_image(floating+ref_list[0],np.uint8,0)
            band_list1 = read_image(ref+floating_list[0],np.uint8,0)
            im0=band_list0[0]
            im1=band_list1[0]       
            a=Extraction(floating+ref_list[0],ref+floating_list[0])# the coordinates of the max point which is supposed to be the invariant point
            a_list.append(a[0][0] - a[0][2])
            a_list.append(a[0][1] - a[0][3])
            b=affine_corrected(im0,a[0][2] - a[0][0],a[0][3] - a[0][1])
            out_list=[]
            out_list.append(b)
            write_image(out_list,0,i,path+'corrected_Transl._'+floating[-11:-1]+'_B'+str(i)+'.tif',rows,cols_q,geotransform,projection)       
    return a_list

def height(shadowShape,pixelWidth,pixelHeight,outShape='',ui_progress=None):
    
    '''Function for calculate and assign to shadows shapefile the height of building and length of shadow.            \
    Build new dataset (or shapefile if outputShape argument declared) with field Shadow_Len and Height which contains \
    respectively the height of relative building and the length of the shadow                                         \
    
    :param shadowShape: path of shadows shapefile (str)
    :param pixelWidth: value in meters of length of 1 pixel (float)
    :param pixelHeight: value in meters of length of 1 pixel (float)
    :param outputShape: path of output shapefile (char)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
    shadowDS = driver.Open(shadowShape)
    pixelWidth = pixelWidth
    pixelHeight = pixelWidth
    shadowLayer = shadowDS.GetLayer()
    shadowFeaturesCount = shadowLayer.GetFeatureCount()
    if outShape == '':
        driver = osgeo.ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outShape)
    outDS.CopyLayer(shadowLayer,"Shadows")
    outLayer = outDS.GetLayer()
    shadow_def = osgeo.ogr.FieldDefn('Shadow_Len', osgeo.ogr.OFTInteger)
    height_def = osgeo.ogr.FieldDefn('Height',osgeo.ogr.OFTReal)
    outLayer.CreateField(shadow_def)
    outLayer.CreateField(height_def)
    if ui_progress:
        ui_progress.progressBar.setMinimum(1)
        ui_progress.progressBar.setMaximum(shadowFeaturesCount)
    for i in range(0,shadowFeaturesCount):
        #Convert Shape to Raster
        shadowFeature = shadowLayer.GetFeature(i)
        x_min, x_max, y_min, y_max = WindowsMaker(shadowFeature).make_coordinates()
        cols = int(math.ceil(float((x_max-x_min)) / float(pixelWidth))) #definition of the dimensions according to the extent and pixel resolution
        rows = int(math.ceil(float((y_max-y_min)) / float(pixelHeight)))
        rasterDS = gdal.GetDriverByName('MEM').Create("", cols,rows, 1, GDT_UInt16)
        rasterDS.SetGeoTransform((x_min, pixelWidth, 0,y_max, 0, -pixelHeight))
        rasterDS.SetProjection(shadowLayer.GetSpatialRef().ExportToWkt())
        gdal.RasterizeLayer(rasterDS,[1], shadowLayer,burn_values=[1])

        #Make Band List
        inband = rasterDS.GetRasterBand(1) 
        band_list = inband.ReadAsArray().astype(np.uint16)
        #Process Raster with Scipy
        w = ndimage.morphology.binary_fill_holes(band_list)
        band_list = ndimage.binary_opening(w, structure=np.ones((3,3))).astype(np.uint16)
        #Calculate zone
        prj = rasterDS.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        zone = int(str(srs.GetAttrValue('projcs')).split('Zone')[1][1:-1])
        #Calculate shadow Lenght
        geo_transform = rasterDS.GetGeoTransform()
        easting = geo_transform[0]
        northing = geo_transform[3]
        a = utm2wgs84(easting, northing, zone)
        lon,lat = a[0],a[1]
        #lat,lon = northing, easting
        shadowLen = shadow_length(band_list,lat,lon,'2012/8/11 9:35:00')
        buildingHeight = building_height(lat,lon,'2012/8/11 9:35:00',shadowLen)
        print "buildingHeight: {}\nshadowLen: {}".format(buildingHeight,shadowLen)
        #insert value to init shape
        outFeature = outLayer.GetFeature(i)
        outFeature.SetField('Shadow_Len',int(shadowLen))
        outFeature.SetField('Height',buildingHeight)
        outLayer.SetFeature(outFeature)
        if ui_progress:
            ui_progress.progressBar.setValue(ui_progress.progressBar.value()+1)
    return outDS

def shadow_checker(buildingShape, shadowShape, date, idfield="ID", outputShape='', resize=0, ui_progress=None):

    '''Function to assign right shadows to buildings using the azimuth calculated from the raster acquisition date and time.\
    Build new dataset (or shapefile if outputShape argument declared) with field ShadowID and Height related to   \
    the id of relative shadow fileshape and the height of building calculated with sensum.ShadowPosition \
    class. Polygons are taken from the building-related layer. Shadow shapefile has to be just processed with height() function. 

    :param buildingShape: path of buildings shapefile (str)
    :param shadowShape: path of shadows shapefile (str)
    :param date: date of acquisition in EDT (example: '1984/5/30 17:40:56') (char)
    :param idfield: id field of shadows (char)
    :param outputShape: path of output shapefile (char)
    :param resize: resize value for increase dimension of the form expressed in coordinates as unit of measure (float)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    #get layers
    driver = ogr.GetDriverByName("ESRI Shapefile")
    buildingsDS = driver.Open(buildingShape)
    buildingsLayer = buildingsDS.GetLayer()
    buindingsFeaturesCount = buildingsLayer.GetFeatureCount()
    shadowDS = driver.Open(shadowShape)
    shadowLayer = shadowDS.GetLayer()
    if outputShape == '':
        driver = ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outputShape)
    #copy the buildings layer and use it as output layer
    outDS.CopyLayer(buildingsLayer,"")
    outLayer = outDS.GetLayer()
    #add fields 'ShadowID' and 'Height' to outLayer
    fldDef = ogr.FieldDefn('ShadowID', ogr.OFTInteger)
    outLayer.CreateField(fldDef)
    fldDef = ogr.FieldDefn('Height', ogr.OFTReal)
    outLayer.CreateField(fldDef)
    #Calculate zone
    spatialRef = outLayer.GetSpatialRef()
    zone = int(str(spatialRef.GetAttrValue('projcs')).split('Zone')[1][1:-1])
    #get latitude and longitude from 1st features for calculate azimuth
    buildingFeature = outLayer.GetNextFeature()
    buildingGeometry = buildingFeature.GetGeometryRef().Centroid()
    longitude = float(buildingGeometry.GetX())
    latitude = float(buildingGeometry.GetY())
    longitude, latitude, altitude = utm2wgs84(longitude, latitude, zone)
    #calculate azimut and get right operators (> and <) as function used for filter the position of shadow
    position = ShadowPosition(date,latitude=latitude,longitude=longitude)
    position.main()
    xOperator, yOperator = position.operator()
    if ui_progress:
        ui_progress.progressBar.setValue(1)
        ui_progress.progressBar.setMinimum(1)
        ui_progress.progressBar.setMaximum(buindingsFeaturesCount)
    #loop into building features goint to make a window around each features and taking only shadow features there are into the window. 
    for i in range(buindingsFeaturesCount):
        print "{} di {} features".format(i+1,buindingsFeaturesCount)
        buildingFeature = outLayer.GetFeature(i)
        #make a spatial layer and get the layer
        maker = WindowsMaker(buildingFeature,resize)
        maker.make_feature()
        spatialDS = maker.get_shapeDS(shadowLayer)
        spatialLayer = spatialDS.GetLayer()
        spatialLayerFeatureCount = spatialLayer.GetFeatureCount()
        #check if there are any features into the spatial layer
        if spatialLayerFeatureCount:
            spatialFeature = spatialLayer.GetNextFeature()
            outFeature = outLayer.GetFeature(i)
            outGeometry = outFeature.GetGeometryRef().Centroid()
            xOutGeometry = float(outGeometry.GetX())
            yOutGeometry = float(outGeometry.GetY())
            #if the count of features is > 1 need to find the less distant feature
            if spatialLayerFeatureCount > 1:
                fields = []
                centros = []
                heights = []
                #loop into spatial layer
                while spatialFeature:
                    #calc of centroid of spatial feature
                    spatialGeometry = spatialFeature.GetGeometryRef().Centroid()
                    xSpatialGeometry = float(spatialGeometry.GetX())
                    ySpatialGeometry = float(spatialGeometry.GetY())
                    #filter the position of shadow, need to respect the right azimuth
                    if xOperator(xSpatialGeometry,xOutGeometry) and yOperator(ySpatialGeometry,yOutGeometry):
                        #saving id,centroid and heights into lists
                        fields.append(spatialFeature.GetField(idfield))
                        centros.append(spatialFeature.GetGeometryRef().Centroid())
                        heights.append(spatialFeature.GetField('Height'))
                    spatialFeature = spatialLayer.GetNextFeature()
                #calc distance from centroids
                distances = [outGeometry.Distance(centro) for centro in centros]
                #make multilist named datas with: datas[0] = distances, datas[1] = fields, datas[2] = heights and order for distances
                datas = sorted(zip(distances,fields,heights))
                #take less distant
                if datas:
                    outFeature.SetField("ShadowID",datas[0][1])
                    outFeature.SetField("Height",datas[0][2])
                    outLayer.SetFeature(outFeature)
            #spatialFeature = 1 so don't need to check less distant
            else:
                spatialGeometry = spatialFeature.GetGeometryRef().Centroid()
                xSpatialGeometry = float(spatialGeometry.GetX())
                ySpatialGeometry = float(spatialGeometry.GetY())
                if xOperator(xSpatialGeometry,xOutGeometry) and yOperator(ySpatialGeometry,yOutGeometry):
                    field = spatialFeature.GetField(idfield)
                    outFeature.SetField("ShadowID",field)
                    field = spatialFeature.GetField("Height")
                    outFeature.SetField("Height",field)
                    outLayer.SetFeature(outFeature)
                    spatialFeature = spatialLayer.GetNextFeature()
        if ui_progress:
            ui_progress.progressBar.setValue(ui_progress.progressBar.value()+1)
    return outDS

def segmentation_optimizer(input_image,input_shape,segmentation_name,select_criteria,nloops,ui_progress=None):
    band_list = read_image(input_image,np.uint16,0)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
    #Open reference shapefile
    driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
        
    inDS = driver_shape.Open(input_shape, 0)
    if inDS is None:
        print 'Could not open file'
        sys.exit(1)
    inLayer = inDS.GetLayer()
    numFeatures = inLayer.GetFeatureCount()
    print 'Number of reference features: ' + str(numFeatures)
    #temp_shape = input_shape[:-4]+'_temp.shp'
    patches_list = []
    patches_geo_transform_list = []
    reference_list = []
    ref_geo_transform_list = []
    if ui_progress:
        ui_progress.progressBar.setMinimum(1)
        ui_progress.progressBar.setMaximum(numFeatures)
    for n in range(0,numFeatures):
        
        #separate each polygon creating a temp file
        temp = split_shape(inLayer,n)
        
        #conversion of the temp file to raster
        #temp = driver_shape.Open(temp_shape, 0)
        temp_layer = temp.GetLayer()
        
        reference_matrix, ref_geo_transform = polygon2array(temp_layer,geo_transform[1],abs(geo_transform[5])) 
        temp.Destroy()
        #driver_shape.DeleteCtaSource(temp_shape)
        reference_list.append(reference_matrix)
        ref_geo_transform_list.append(ref_geo_transform)
        
        ext_patch_list,patch_geo_transform = create_extended_patch(band_list,reference_matrix,geo_transform,ref_geo_transform,0.3,False)
        patches_list.append(ext_patch_list)
        patches_geo_transform_list.append(patch_geo_transform)
        if ui_progress:
            ui_progress.progressBar.setValue(ui_progress.progressBar.value()+1)
        
    e = call_optimizer(segmentation_name,patches_list,reference_list,patches_geo_transform_list,ref_geo_transform_list,projection,select_criteria,nloops)
    return e

def test_features(input_file,segmentation_shape,output_shape,indexes_list_spectral,indexes_list_texture,field="DN",ui_progress=None):

    #indexes_list_spectral = ['mean','mode','std','ndvi_mean','ndvi_std'] #possible values: 'mean', 'mode', 'std', 'max_br', 'min_br', 'ndvi_mean', 'ndvi_std', 'weigh_br'
    #indexes_list_texture = [] #possible values: 'contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM'

    ############################################################################################################################

    start_time = time.time()
    ndvi_comp = []
    wb_comp = []
    #Read original image - base layer
    input_list = read_image(input_file,np.uint16,0)
    #input_list_tf = read_image(input_file,np.uint8,0) #different data type necessary for texture features
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_file)
    input_list_tf = linear_quantization(input_list,64)
    #Conversion of the provided segmentation shapefile to raster for further processing
    shp2rast(segmentation_shape, segmentation_shape[:-4]+'.TIF', rows, cols, field)
    seg_list = read_image(segmentation_shape[:-4]+'.TIF',np.int32,0)

    if (('ndvi_mean' in indexes_list_spectral) or ('ndvi_std' in indexes_list_spectral)) and nbands > 3:
        ndvi = (input_list[3]-input_list[2]) / (input_list[3]+input_list[2]+0.000001)
        ndvi_comp = [s for s in indexes_list_spectral if 'ndvi_mean' in s or 'ndvi_std' in s]

    if 'weigh_br' in indexes_list_spectral:
        band_sum = np.zeros((rows,cols))
        for b in range(0,nbands):
            band_sum = band_sum + input_list[b]
        wb_comp = [s for s in indexes_list_spectral if 'weigh_br' in s]
        
    ind_list_spectral = [s for s in indexes_list_spectral if 'ndvi_mean' not in s or 'ndvi_std' not in s or 'weigh_br' not in s]
    print ind_list_spectral
    #read input shapefile
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile=driver_shape.Open(segmentation_shape,0)
    inlayer=infile.GetLayer()

    #create output shapefile
    if os.path.isfile(output_shape):
        os.remove(output_shape)
    outfile=driver_shape.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Features',geom_type=osgeo.ogr.wkbPolygon)

    layer_defn = inlayer.GetLayerDefn()
    infeature = inlayer.GetNextFeature()

    dn_def = osgeo.ogr.FieldDefn(field, osgeo.ogr.OFTInteger)
    outlayer.CreateField(dn_def)
    #max_brightness, min_brightness, ndvi_mean, ndvi_standard_deviation, weighted_brightness
    for b in range(1,nbands+1):
        for si in range(0,len(ind_list_spectral)):
            field_def = osgeo.ogr.FieldDefn(ind_list_spectral[si] + str(b), osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
        if ndvi_comp:
            for nd in range(0,len(ndvi_comp)):
                field_def = osgeo.ogr.FieldDefn(ndvi_comp[nd] + str(b), osgeo.ogr.OFTReal)
                outlayer.CreateField(field_def)
        if wb_comp:
            field_def = osgeo.ogr.FieldDefn(wb_comp[0] + str(b), osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
        for sp in range(0,len(indexes_list_texture)):
            if len(indexes_list_texture[sp]+str(b)) > 10:
                cut = len(indexes_list_texture[sp]+str(b)) - 10 
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp][:-cut] + str(b), osgeo.ogr.OFTReal)
            else:
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp] + str(b), osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
          
    feature_def = outlayer.GetLayerDefn()
    n_feature = inlayer.GetFeatureCount()
    i = 1

    #loop through segments
    if ui_progress:
        ui_progress.progressBar.setMinimum(1)
        ui_progress.progressBar.setMaximum(n_feature)
    while infeature:
        print str(i) + ' of ' + str(n_feature)
        i = i+1
        # get the input geometry
        geom = infeature.GetGeometryRef()
        # create a new feature
        outfeature = osgeo.ogr.Feature(feature_def)
        # set the geometry and attribute
        outfeature.SetGeometry(geom)
        #field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        dn = infeature.GetField(field)
        outfeature.SetField(field,dn)
     
        for b in range(1,nbands+1):
            
            spectral_list = spectral_segments(input_list[b-1], dn, seg_list[0], ind_list_spectral, nbands)
            for si in range(0,len(indexes_list_spectral)):
                outfeature.SetField(indexes_list_spectral[si] + str(b),spectral_list[si])
            
            texture_list = texture_segments(input_list_tf[b-1],dn,seg_list[0],indexes_list_texture)
            for sp in range(0,len(indexes_list_texture)):
                if len(indexes_list_texture[sp]+str(b)) > 10:
                    cut = len(indexes_list_texture[sp]+str(b)) - 10
                    outfeature.SetField(indexes_list_texture[sp][:-cut] + str(b),texture_list[sp])
                else:
                    outfeature.SetField(indexes_list_texture[sp] + str(b),texture_list[sp])
            if ndvi_comp:
                ndvi_list = spectral_segments(ndvi, dn, seg_list[0], ndvi_comp, nbands)
                for nd in range(0,len(ndvi_comp)):
                    outfeature.SetField(ndvi_comp[nd] + str(b),ndvi_list[nd])
            if wb_comp:
                wb = spectral_segments(band_sum, dn, seg_list[0], wb_comp, nbands)
                outfeature.SetField(wb_comp[0] + str(b),wb[0])
                
        outlayer.CreateFeature(outfeature)
        outfeature.Destroy()
        infeature = inlayer.GetNextFeature()
        if ui_progress:
            ui_progress.progressBar.setValue(ui_progress.progressBar.value()+1)

    shutil.copyfile(segmentation_shape[:-4]+'.prj', output_shape[:-4]+'.prj')
    print 'Output created: ' + output_shape
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()

    end_time = time.time()
    print 'Total time = ' + str(end_time-start_time)    

class Sensum:

    def __init__(self, iface):
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        localePath = os.path.join(self.plugin_dir, 'i18n', 'sensum_{}.qm'.format(locale))

        if os.path.exists(localePath):
            self.translator = QTranslator()
            self.translator.load(localePath)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

    def initGui(self):

        self.toolBar = self.iface.addToolBar("SENSUM")

        ######################
        ## PANSHARP
        ######################
        # Create action that will start plugin configuration
        self.action_pansharp = QAction(
            QIcon(":/plugins/sensum_plugin/icons/pansharp.png"),
            u"Pansharp", self.iface.mainWindow())
        # connect the action to the run method
        self.action_pansharp.triggered.connect(self.pansharp)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_pansharp)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_pansharp)
        
        ######################
        ## CLASSIFICATION
        ######################
        # Create action that will start plugin configuration
        self.action_classification = QAction(
            QIcon(":/plugins/sensum_plugin/icons/classification.png"),
            u"Classification", self.iface.mainWindow())
        # connect the action to the run method
        self.action_classification.triggered.connect(self.classification)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_classification)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_classification)

        ######################
        ## SEGMENTATION
        ######################
        # Create action that will start plugin configuration
        self.action_segmentation = QAction(
            QIcon(":/plugins/sensum_plugin/icons/segmentation.png"),
            u"Segmentation", self.iface.mainWindow())
        # connect the action to the run method
        self.action_segmentation.triggered.connect(self.segmentation)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_segmentation)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_segmentation)

        ######################
        ## FEATURES
        ######################
        # Create action that will start plugin configuration
        self.action_features = QAction(
            QIcon(":/plugins/sensum_plugin/icons/features.png"),
            u"Features", self.iface.mainWindow())
        # connect the action to the run method
        self.action_features.triggered.connect(self.features)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_features)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_features)

        ######################
        ## BUILDING HEIGHT
        ######################
        # Create action that will start plugin configuration
        self.action_building_height = QAction(
            QIcon(":/plugins/sensum_plugin/icons/buildingsheight.png"),
            u"Building Height", self.iface.mainWindow())
        # connect the action to the run method
        self.action_building_height.triggered.connect(self.build_height)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_building_height)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_building_height)

        ######################
        ## CO-REGISTRATION
        ######################
        # Create action that will start plugin configuration
        self.action_coregistration = QAction(
            QIcon(":/plugins/sensum_plugin/icons/coregistration.png"),
            u"Coregistration", self.iface.mainWindow())
        # connect the action to the run method
        self.action_coregistration.triggered.connect(self.coregistration)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_coregistration)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_coregistration)

        ######################
        ## FOOTPRINTS
        ######################
        # Create action that will start plugin configuration
        self.action_footprints = QAction(
            QIcon(":/plugins/sensum_plugin/icons/footprints.png"),
            u"Footprints", self.iface.mainWindow())
        # connect the action to the run method
        self.action_footprints.triggered.connect(self.footprints)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_footprints)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_footprints)

        ######################
        ## STACK SATELLITE
        ######################
        # Create action that will start plugin configuration
        self.action_stacksatellite = QAction(
            QIcon(":/plugins/sensum_plugin/icons/stacksatellite.png"),
            u"Stack Satellite", self.iface.mainWindow())
        # connect the action to the run method
        self.action_stacksatellite.triggered.connect(self.stacksatellite)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_stacksatellite)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_stacksatellite)

    def unload(self):

        self.iface.removePluginMenu(u"&SENSUM", self.action_pansharp)
        self.iface.removeToolBarIcon(self.action_pansharp)
        self.iface.removePluginMenu(u"&SENSUM", self.action_classification)
        self.iface.removeToolBarIcon(self.action_classification)
        self.iface.removePluginMenu(u"&SENSUM", self.action_segmentation)
        self.iface.removeToolBarIcon(self.action_segmentation)
        self.iface.removePluginMenu(u"&SENSUM", self.action_features)
        self.iface.removeToolBarIcon(self.action_features)
        self.iface.removePluginMenu(u"&SENSUM", self.action_building_height)
        self.iface.removeToolBarIcon(self.action_building_height)
        self.iface.removePluginMenu(u"&SENSUM", self.action_coregistration)
        self.iface.removeToolBarIcon(self.action_coregistration)
        self.iface.removePluginMenu(u"&SENSUM", self.action_footprints)
        self.iface.removeToolBarIcon(self.action_footprints)
        self.iface.removePluginMenu(u"&SENSUM", self.action_stacksatellite)
        self.iface.removeToolBarIcon(self.action_stacksatellite)

    # run method that performs all the real work
    def pansharp(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = PansharpDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            multiband_image = str(ui.lineEdit_multiband.text())
            panchromatic_image = str(ui.lineEdit_panchromatic.text())
            output_image = str(ui.lineEdit_output.text())
            pansharp(multiband_image,panchromatic_image,output_image)
            QMessageBox.information(None, "Info", 'Done!')

    def classification(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = ClassificationDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            input_image = str(ui.lineEdit_input.text())
            output_image = str(ui.lineEdit_output.text())
            input_training_field = str(ui.lineEdit_training_field.text())
            input_classification_type = str(ui.comboBox_supervised.currentText())
            if input_classification_type == "Supervised":
                input_training_image = str(ui.lineEdit_training.text())
                input_classification_supervised_type = str(ui.comboBox_supervised_type.currentText())
                training_field = str(ui.lineEdit_training_field.text())
                train_classifier_otb([input_image],[input_training_image],"tmp.txt",input_classification_supervised_type,training_field)
                supervised_classification_otb(input_image,"tmp.txt",output_image)
            else:
                n_classes = int(ui.spinBox_nclasses.text())
                n_iterations = int(ui.spinBox_niteration.text())
                unsupervised_classification_otb(input_image,output_image,n_classes,n_iterations)
            QMessageBox.information(None, "Info", 'Done!')

    def segmentation(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = SegmentationDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            input_image = str(ui.lineEdit_input.text())
            output_image = str(ui.lineEdit_output.text())
            checked_optimizer = bool(ui.checkBox_optimizer.isChecked())
            segm_mode = str(ui.comboBox_method.currentText())
            if checked_optimizer:
                optimizer_shape = str(ui.lineEdit_optimizer_input.text())
                nloops = int(ui.spinBox_nloops.text())
                select_criteria = int(ui.spinBox_criteria.text())
                floaters = bool(ui.radioButton_floaters.isChecked())
                segm_mode_optimazer = segm_mode
                if floaters and segm_mode == "Baatz":
                    segm_mode_optimazer = "Baatz"
                elif floaters == 0 and segm_mode == "Baatz":
                    segm_mode_optimazer = "Baatz_integers"
                elif floaters and segm_mode == "Region Growing":
                    segm_mode_optimazer = "Region_growing"
                elif floaters == 0 and segm_mode == "Region Growing":
                    segm_mode_optimazer = "Region_growing_integers"
                elif segm_mode == "Morphological Profiles":
                    segm_mode_optimazer = "Mprofiles"
                e = segmentation_optimizer(input_image,optimizer_shape,segm_mode_optimazer,select_criteria,nloops)
                if segm_mode == 'Felzenszwalb':
                    input_band_list = read_image(input_image,0,0)
                    felzenszwalb_skimage(input_band_list, float(e[0]), float(e[1]), 0)
                if segm_mode == 'Edison':
                    edison_otb(input_image,"vector",output_image,int(round(e[0])),int(round(e[1])),0,0)
                if segm_mode == 'Meanshift':
                    meanshift_otb(input_image,output_image,'vector',int(round(e[0])),float(e[1]),0,0,0)
                if segm_mode == 'Watershed':
                    watershed_otb(input_image,'vector',output_image,0,float(e[0]))
                if segm_mode == 'Morphological Profiles':
                    mprofiles_otb(input_image,output_image,'vector',0,int(round(e[0])),0,0)
                if segm_mode == 'Baatz':
                    if floaters:
                        segments_baatz = baatz_interimage(input_image,0,float(e[0]),float(e[1]),0,True)
                    else:    
                        segments_baatz = baatz_interimage(input_image,int(round(e[0])),Compact,Color,int(round(e[1])),True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_baatz],0,0,output_image[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_image[:-4]+'.TIF',output_image)
                if segm_mode == 'Region_growing':
                    if floaters:
                        segments_regiongrowing = region_growing_interimage(input_image,int(round(e[0])),0,0,int(round(e[1])),True)
                    else:
                        segments_regiongrowing = region_growing_interimage(input_image,EuclideanT,float(e[0]),float(e[1]),Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_regiongrowing],0,0,output_image[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_image[:-4]+'.TIF',output_image)
            else:
                if segm_mode == "Felzenszwalb":
                    min_size = float(ui.lineEdit_felzenszwalb_minsize.text())
                    scale = float(ui.lineEdit_felzenszwalb_scale.text())
                    sigma = int(ui.lineEdit_felzenszwalb_sigma.text())
                    input_band_list = read_image(input_image,0,0)
                    felzenszwalb_skimage(input_band_list, scale, sigma, min_size)
                if segm_mode == "Edison":
                    spatial_radius = int(ui.lineEdit_edison_radius.text())
                    range_radius = int(ui.lineEdit_edison_range.text())
                    min_size = int(ui.lineEdit_edison_size.text())
                    scale = int(ui.lineEdit_edison_scale.text())
                    edison_otb(input_image,"vector",output_image,spatial_radius,range_radius,min_size,scale)
                elif segm_mode == "Baatz":
                    EuclideanT = int(ui.lineEdit_baatz_euclidean.text())
                    Compact = float(ui.lineEdit_baatz_compactness.text())
                    Color = float(ui.lineEdit_baatz_color.text())
                    Scale = int(ui.lineEdit_baatz_scale.text())
                    segments_baatz = baatz_interimage(input_image,EuclideanT,Compact,Color,Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_baatz],0,0,output_image[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_image[:-4]+'.TIF',output_image)
                elif segm_mode == "Meanshift":
                    SpatialR = int(ui.lineEdit_meanshift_spatial.text())
                    RangeR = float(ui.lineEdit_meanshift_range.text())
                    Thres = float(ui.lineEdit_meanshift_threshold.text())
                    MaxIter = int(ui.lineEdit_meanshift_iterations.text())
                    MinimumS = int(ui.lineEdit_meanshift_minsize.text())
                    meanshift_otb(input_image,output_image,'vector',SpatialR,RangeR,Thres,MaxIter,MinimumS)
                elif segm_mode == "Watershed":
                    Thres = float(ui.lineEdit_watershed_threshold.text())
                    Level = float(ui.lineEdit_watershed_level.text())
                    watershed_otb(input_image,'vector',output_image,Thres,Level)
                elif segm_mode == "Morphological Profiles":
                    Size = int(ui.lineEdit_morphological_size.text())
                    Sigma = float(ui.lineEdit_morphological_sigma.text())
                    Start = float(ui.lineEdit_morphological_start.text())
                    Step = int(ui.lineEdit_morphological_step.text())
                    mprofiles_otb(input_image,output_image,'vector',Size,Sigma,Start,Step)
                elif segm_mode == "Region Growing":
                    EuclideanT = int(ui.lineEdit_region_euclidean.text())
                    Compact = float(ui.lineEdit_region_compactness.text())
                    Color = float(ui.lineEdit_region_color.text())
                    Scale = int(ui.lineEdit_region_scale.text())
                    segments_regiongrowing = region_growing_interimage(input_image,EuclideanT,Compact,Color,Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_regiongrowing],0,0,output_image[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_image[:-4]+'.TIF',output_image)
            QMessageBox.information(None, "Info", 'Done!')

    def features(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = FeaturesDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            input_image = str(ui.lineEdit_input.text())
            input_shape = str(ui.lineEdit_training.text())
            output_shape = str(ui.lineEdit_output.text())
            field = str(ui.lineEdit_field.text())
            spectrals = [(ui.checkBox_maxbr,"max_br"),(ui.checkBox_mean,"mean"),(ui.checkBox_minbr,"min_br"),(ui.checkBox_mode,"mode"),(ui.checkBox_ndivistd,"ndvi_std"),(ui.checkBox_ndvimean,"ndvi_mean"),(ui.checkBox_std,"std"),(ui.checkBox_weighbr,"weigh_br")]
            textures = [(ui.checkBox_asm,"ASM"),(ui.checkBox_contrast,"contrast"),(ui.checkBox_correlation,"correlation"),(ui.checkBox_dissimilarity,"dissimilarity"),(ui.checkBox_energy,"energy"),(ui.checkBox_homogeneity,"homogeneity")]
            indexes_list_spectral = [index for pushButton,index in spectrals if pushButton.isChecked()]
            indexes_list_texture = [index for pushButton,index in textures if pushButton.isChecked()]
            test_features(input_image,input_shape,output_shape,indexes_list_spectral,indexes_list_texture,field=field,ui_progress=dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')

    def build_height(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = BuildHeightDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            input_buildings = str(ui.lineEdit_input_buildings.text())
            input_shadow = str(ui.lineEdit_input_shadows.text())
            output_shape = str(ui.lineEdit_output.text())
            idfield = str(ui.lineEdit_shadow_field.text())
            window_resize = float(ui.doubleSpinBox_window_paramater.text())
            date = str(ui.dateTimeEdit.text())
            #tmp_shadow_processed = tempfile.mkstemp()[1]
            #os.remove(tmp_shadow_processed)
            if os.path.isfile(output_shape):
                os.remove(output_shape)
            #height(input_shadow,0.5,0.5,outShape=tmp_shadow_processed,ui_progress=dlgProgress.ui)
            #shadow_checker(input_buildings,tmp_shadow_processed, date, outputShape=output_shape, idfield=idfield, resize=window_resize,ui_progress=dlgProgress.ui)
            print input_shadow,input_buildings,date,output_shape,idfield,window_resize
            if os.name == "posix": 
                os.system("{}/height.py \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"".format(os.path.dirname(os.path.abspath(__file__)),input_shadow,input_buildings,date,output_shape,idfield,window_resize))
            else:
                os.system("python.exe {}/height.py \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"".format(os.path.dirname(os.path.abspath(__file__)),input_shadow,input_buildings,date,output_shape,idfield,window_resize))
            #shutil.rmtree(tmp_shadow_processed)
            QMessageBox.information(None, "Info", 'Done!')

    def coregistration(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = CoregistrationDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            #if os.path.isfile(output_shape):
                #os.remove(output_shape)
            path = str(ui.lineEdit_tobechange.text())+"/"
            ref = str(ui.lineEdit_reference.text())+"/"
            #path= '/home/gale/Bishkek_daniel/'
            #ref=path+'2010-07-27/'#reference directory
            sat_folder = os.listdir(path)
            sat_folder = [s for s in sat_folder if "-"in s ]
            #print sat_folder
            import glob
            floating_list=[]
            for i in sat_folder:
                floating=path+i+'/'
                floating_list.append(floating)

            for j in floating_list:
                starttime=time.time()
                #print starttime
                print F_B(path,j,ref)  
                endtime=time.time()
                time_total = endtime-starttime
                print time_total
            QMessageBox.information(None, "Info", 'Done!')

    def footprints(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = FootprintsDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            pansharp_file = str(ui.lineEdit_pansharp.text())
            training_set = str(ui.lineEdit_training.text())
            training_attribute = str(ui.lineEdit_training_field.text())
            output_shape = str(ui.lineEdit_output.text())
            checked_optimizer = bool(ui.checkBox_filter.isChecked())
            building_classes = [str(ui.listWidget.item(index).text()) for index in xrange(ui.listWidget.count())]
            footprints(pansharp_file,training_set,training_attribute,building_classes,output_shape,enable_smooth_filter=checked_optimizer,ui_progress=dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')

    def stacksatellite(self):
        # Create the dialog (after translation) and keep reference
        self.dlg = StackSatelliteDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg.ui
            dlgProgress.show()
            sat_folder = str(ui.lineEdit_satellite_folder.text())
            input_shapefile = str(ui.lineEdit_input_shapefile.text())
            quantization_mode = str(ui.comboBox_quantization_mode.currentText())
            opt_polygon = str(ui.lineEdit_optimazer_polygon.text())
            segmentation_name = str(ui.comboBox_segmentation.currentText())
            select_criteria = str(ui.spinBox_select_criteria.text())
            nloops = str(ui.spinBox_niteration.text())
            n_classes = str(ui.spinBox_nclasses.text())
            ref_dir = (str(ui.lineEdit_reference_directory.text() if bool(ui.checkBox_restrict_to_city.isChecked()) else ""))
            restrict_to_city = bool(ui.checkBox_restrict_to_city.isChecked())
            coregistration = bool(ui.checkBox_coregistration.isChecked())
            builtup_index_method = bool(ui.checkBox_builtup_index.isChecked())
            pca_index_method = bool(ui.checkBox_pca_index.isChecked())
            pca_classification_method = bool(ui.checkBox_pca_classification.isChecked())
            dissimilarity_method = bool(ui.checkBox_dissimilarity.isChecked())
            band_combination = bool(ui.checkBox_band_combination.isChecked())
            supervised_method = bool(ui.checkBox_supervised.isChecked())
            unsupervised_method = bool(ui.checkBox_unsupervised.isChecked())
            supervised_polygon = bool(ui.checkBox_unsupervised.isChecked())
            stacksatellite(sat_folder, input_shapefile, quantization_mode, opt_polygon, segmentation_name, select_criteria, nloops, n_classes, ref_dir, restrict_to_city, coregistration, builtup_index_method, pca_index_method, pca_classification_method, dissimilarity_method, band_combination, supervised_method, supervised_polygon)
            QMessageBox.information(None, "Info", 'Done!')
