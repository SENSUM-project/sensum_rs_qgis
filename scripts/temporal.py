#!/usr/bin/python
import config
import os,sys
import shutil
import time
import tempfile
import gdal
import ogr
from gdalconst import *
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

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    sat_folder = str(arg.sat_folder)
    input_shapefile = str(arg.input_shapefile)
    n_classes = int(arg.n_classes)
    if bool(arg.plot):
        indexes_list = map(str, arg.indexes)
        print indexes_list
        graph(sat_folder, indexes_list)
    else:
        temporal(input_shapefile, sat_folder, n_classes)

def args():
    parser = argparse.ArgumentParser(description='Temporal Analysis')
    parser.add_argument("sat_folder", help="????")
    parser.add_argument("input_shapefile", help="????")
    parser.add_argument("n_classes", help="????")
    parser.add_argument("-i", "--indexes", nargs="+", help="????")
    parser.add_argument("--plot", default=False, const=True, nargs='?', help="????")
    args = parser.parse_args()
    return args

def temporal(input_shapefile, sat_folder, n_classes):
    indexes_list = ["Index1","Index2","Index3","Index4","Index5","Index6","Index7","Index8","Index9","Index10","Index11","Index12"]
    ref_dir = None
    restrict_to_city = True
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'
    sat_folder = sat_folder + separator     
    dirs = os.listdir(sat_folder)
    band_list = []
    cd_names = []
    c = 0
    dirs.sort()
    dirs = [dir for dir in dirs if os.path.isdir(sat_folder+dir)]
    if ref_dir is None or ref_dir == '': ref_dir = sat_folder+dirs[-1]+separator
    ref_files = os.listdir(ref_dir)
    if restrict_to_city == True: #Clip the original images with the provided shapefile
        ref_list = [s for s in ref_files if ".TIF" in s and not "_city" in s and "aux.xml" not in s] #look for original landsat files
        for j in range(0,len(ref_list)):
            clip_rectangular(ref_dir+ref_list[j],data_type,input_shapefile,ref_dir+ref_list[j][:-4]+'_city.TIF',mask=True)
        ref_files = os.listdir(ref_dir)
        ref_list = [s for s in ref_files if "_city.TIF" in s and "aux.xml" not in s]
    else: 
        ref_list = [s for s in ref_files if ".TIF" in s and "aux.xml" not in s]
    for n in range(0,len(ref_list)):
        band_ref = read_image(ref_dir+ref_list[n],data_type,0)
        band_list.append(band_ref[0])
       
    rows_ref,cols_ref,nbands_ref,geo_transform_ref,projection_ref = read_image_parameters(ref_dir+ref_list[0])
    if len(band_list) < 10:
        band_list = normalize_to_L8(band_list)
    elif len(band_list) > 10:
        new_list = (band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9])
        band_list = new_list
    change_detection_list = band_calculation(band_list, indexes_list)
    '''
    features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
    features_list[3] = features_list[3]*1000
    features_list[4] = features_list[4]*1000
    mask_water = np.less(features_list[3],features_list[2]) #exclude water
    cd_list_masked = []
    for mw in range(0,len(change_detection_list)):
        cd_list_masked.append(np.choose(mask_water[0],(0,change_detection_list[mw])))
    '''
    write_image(change_detection_list,np.float32,0,ref_dir+'change_detection.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
    cd_names.append(ref_dir+'change_detection.TIF')
    #unsupervised_classification_otb(ref_dir+'change_detection.TIF',ref_dir+'change_detection_classification.TIF',5,1000)

    target_directories = list(sat_folder+directory+separator for directory in reversed(dirs) if not os.path.isfile(sat_folder+directory) and (ref_dir!=sat_folder+directory+separator))
    status = Bar(len(target_directories))
    for target_index,target_dir in enumerate(target_directories):
        status(target_index+2)
        band_list = []
        img_files = os.listdir(target_dir)
        if restrict_to_city == True: #Clip the original images with the provided shapefile
            target_list = [s for s in img_files if ".TIF" in s and not "_city" in s] #look for original landsat files
            
            for j in range(0,len(target_list)):
                clip_rectangular(target_dir+target_list[j],data_type,input_shapefile,target_dir+target_list[j][:-4]+'_city.TIF',mask=True)
            target_files = os.listdir(target_dir)
            target_list = [s for s in target_files if "_city.TIF" in s and "aux.xml" not in s]
        else:
            target_files = os.listdir(target_dir) 
            target_list = [s for s in target_files if ".TIF" in s and "aux.xml" not in s]
        rows,cols,nbands,geo_transform,projection = read_image_parameters(target_dir+target_list[0])
        for n in range(0,len(target_list)):
            band_target = read_image(target_dir+target_list[n],data_type,0)
            band_list.append(band_target[0])
            
        if len(band_list) < 10:
            band_list = normalize_to_L8(band_list)
        elif len(band_list) > 10:
            new_list = (band_list[3],band_list[4],band_list[5],band_list[6],band_list[7],band_list[0],band_list[1],band_list[8],band_list[9])
            band_list = new_list
        rows_target,cols_target,nbands_target,geo_transform_target,projection_target = read_image_parameters(target_dir+target_list[0])
        change_detection_list = band_calculation(band_list, indexes_list)
        '''
        features_list = band_calculation(band_list,['SAVI','NDVI','NDBI','MNDWI','BUILT_UP']) #extract indexes
        features_list[3] = features_list[3]*1000
        features_list[4] = features_list[4]*1000
        mask_water = np.less(features_list[3],features_list[2]) #exclude water
        cd_list_masked = []
        for mw in range(0,len(change_detection_list)):
            cd_list_masked.append(np.choose(mask_water[0],(0,change_detection_list[mw])))
        '''
        write_image(change_detection_list,np.float32,0,target_dir+'change_detection.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
        cd_names.append(target_dir+'change_detection.TIF')
        #unsupervised_classification_otb(target_dir+'change_detection.TIF',target_dir+'change_detection_classification.TIF',n_classes,10)

    big_list = []
    output_cd = sat_folder + 'change_detection_all.TIF'
    for k in range(0,len(cd_names)):
        b_list = read_image(cd_names[k],np.float32,0)
        rows,cols,nbands,geo_transform,projection = read_image_parameters(cd_names[k])
        for b in range(0,len(b_list)):
            big_list.append(b_list[b])
    write_image(big_list,np.float32,0,output_cd,rows,cols,geo_transform,projection)    
    unsupervised_classification_otb(output_cd,output_cd[:-4]+'_class.TIF',n_classes,1000)
    band_list_unsup = read_image(output_cd[:-4]+'_class.TIF',np.uint8,0)
    band_list_unsup[0] = band_list_unsup[0]+np.ones(band_list_unsup[0].shape)
    write_image(band_list_unsup,np.uint8,0,output_cd[:-4]+'_reclass.TIF',rows,cols,geo_transform,projection)
    clip_rectangular(output_cd[:-4]+'_reclass.TIF',np.uint8,input_shapefile,output_cd[:-4]+'_reclass_clip.TIF',mask=True)
    

import matplotlib.pyplot as plt
def graph(sat_folder, indexes_list):
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'
    sat_folder = sat_folder + separator     
    dirs = os.listdir(sat_folder)
    dirs.sort()
    dirs = [dir for dir in dirs if os.path.isdir(sat_folder+dir)]
    output = []
    target_directories = list(sat_folder+directory+separator for directory in reversed(dirs) if not os.path.isfile(sat_folder+directory))
    for target_index,target_dir in enumerate(target_directories):
        output.append(classification_statistics(sat_folder+"change_detection_all_class.TIF",target_dir+'change_detection.TIF',indexes_list))

    # red dashes, blue squares and green triangles
    #indexes_list = (int(indexes_list[0].replace("Index",""))-1)
    #print indexes_list
    #result = np.array(output)[:,indexes_list].T
    #print result
    '''
    for n,res in enumerate(result):
        plt.plot(res,label="Class"+str(n+1))
    plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    '''

    plt.plot(output)
    leg_list = [p for p in plt.plot(output)]
    leg_labels = ['Class ' + str(n+1) for n in range(0,len(leg_list))]
    plt.legend(leg_list, leg_labels,bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def classification_statistics(input_raster_classification,input_raster,input_index):

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

    band_list = read_image(input_raster,np.float32,0)
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)

    index = int(input_index[0].replace("Index",""))
    max_class = np.max(band_list_classification[0])
    output = []

    '''
    for b in range(nbands):
        output_band = []
        for value in range(max_class+1):
            mask = np.equal(band_list_classification[0],value)
            data = np.extract(mask,band_list[b])
            data_flat = data.flatten()
            mean_value = np.mean(data_flat)
            output_band.append(mean_value)
        output.append(output_band)
    return output
    '''

    for value in range(max_class+1):
        mask = np.equal(band_list_classification[0],value)
        data = np.extract(mask,band_list[index-1])
        data_flat = data.flatten()
        mean_value = np.mean(data_flat)
        output.append(mean_value)
    return output
        
if __name__ == "__main__":
    main()
