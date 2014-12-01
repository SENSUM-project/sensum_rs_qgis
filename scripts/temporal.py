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

def unsupervised_classification_otb(input_raster,output_raster,n_classes,n_iterations):
    KMeansClassification = otbApplication.Registry.CreateApplication("KMeansClassification") 
    # The following lines set all the application parameters: 
    KMeansClassification.SetParameterString("in", input_raster) 
    KMeansClassification.SetParameterInt("ts", 10000) 
    KMeansClassification.SetParameterInt("nc", n_classes) 
    KMeansClassification.SetParameterInt("maxit", n_iterations) 
    KMeansClassification.SetParameterFloat("ct", 0.0001) 
    KMeansClassification.SetParameterString("out", output_raster) 
    
    # The following line execute the application 
    KMeansClassification.ExecuteAndWriteOutput()

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    sat_folder = str(arg.sat_folder)
    input_shapefile = str(arg.input_shapefile)
    n_classes = int(arg.n_classes)
    indexes_list = map(str, arg.indexes)
    if bool(arg.plot):
        graph(sat_folder, indexes_list, input_shapefile)
    else:
        temporal(input_shapefile, sat_folder, n_classes, indexes_list)

def args():
    parser = argparse.ArgumentParser(description='Temporal Analysis')
    parser.add_argument("sat_folder", help="????")
    parser.add_argument("input_shapefile", help="????")
    parser.add_argument("n_classes", help="????")
    parser.add_argument("-i", "--indexes", nargs="+", help="????")
    parser.add_argument("--plot", default=False, const=True, nargs="?", help="????")
    args = parser.parse_args()
    return args

def temporal(input_shapefile, sat_folder, n_classes,indexes_list):
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
    cd_names_plot = []
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
    write_image(change_detection_list,np.float32,0,ref_dir+'change_detection.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
    cd_names.append(ref_dir+'change_detection.TIF')

    indexes_list_plot = ["Index1","Index2","Index3","Index4","Index5","Index6","Index7","Index8","Index9","Index10","Index11","Index12"]
    change_detection_list_plot = band_calculation(band_list, indexes_list_plot)
    write_image(change_detection_list_plot,np.float32,0,ref_dir+'change_detection_plot.TIF',rows_ref,cols_ref,geo_transform_ref,projection_ref) #write built-up index to file
    cd_names_plot.append(ref_dir+'change_detection_plot.TIF')

    target_directories = list(sat_folder+directory+separator for directory in reversed(dirs) if not os.path.isfile(sat_folder+directory) and (ref_dir!=sat_folder+directory+separator))
    status = Bar(len(target_directories)+1)
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
        write_image(change_detection_list,np.float32,0,target_dir+'change_detection.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
        cd_names.append(target_dir+'change_detection.TIF')
        change_detection_list_plot = band_calculation(band_list, indexes_list_plot)
        write_image(change_detection_list_plot,np.float32,0,target_dir+'change_detection_plot.TIF',rows_target,cols_target,geo_transform_target,projection_target) #write built-up index to file
        cd_names_plot.append(target_dir+'change_detection_plot.TIF')

    big_list = []
    rows_max,cols_max = 0,0
    output_cd = sat_folder + 'change_detection_all.TIF'
    for k in range(0,len(cd_names)):
        rows,cols,nbands,geo_transform,projection = read_image_parameters(cd_names[k])
        if rows > rows_max: rows_max = rows
        if cols > cols_max: cols_max = cols
    print 'Rows max: ' + str(rows_max)
    print 'Cols max: ' + str(cols_max)

    for k in range(0,len(cd_names)):
        b_list = read_image(cd_names[k],np.float32,0)
        rows,cols,nbands,geo_transform,projection = read_image_parameters(cd_names[k])
        for b in range(0,len(b_list)):
            if rows < rows_max:
                diff = rows_max - rows
                b_list[b] = np.vstack((b_list[b],np.zeros((diff,cols))))
            rows,cols = b_list[b].shape
            if cols < cols_max:
                diff = cols_max - cols
                b_list[b] = np.hstack((b_list[b],np.zeros((rows,diff))))
            rows,cols = b_list[b].shape
            big_list.append(b_list[b])
    write_image(big_list,np.float32,0,output_cd,rows_max,cols_max,geo_transform,projection)    
    unsupervised_classification_otb(output_cd,output_cd[:-4]+'_class.TIF',n_classes,100000)
    band_list_unsup = read_image(output_cd[:-4]+'_class.TIF',np.uint8,0)
    band_list_unsup[0] = band_list_unsup[0]+np.ones(band_list_unsup[0].shape)
    write_image(band_list_unsup,np.uint8,0,output_cd[:-4]+'_reclass.TIF',rows_max,cols_max,geo_transform,projection)
    clip_rectangular(output_cd[:-4]+'_reclass.TIF',np.uint8,input_shapefile,output_cd[:-4]+'_reclass_clip.TIF',mask=True)

    big_list_plot = []
    output_cd_plot = sat_folder + 'change_detection_all_plot.TIF'
    for k in range(0,len(cd_names_plot)):
        b_list = read_image(cd_names_plot[k],np.float32,0)
        rows,cols,nbands,geo_transform,projection = read_image_parameters(cd_names_plot[k])
        for b in range(0,len(b_list)):
            if rows < rows_max:
                diff = rows_max - rows
                b_list[b] = np.vstack((b_list[b],np.zeros((diff,cols))))
            rows,cols = b_list[b].shape
            if cols < cols_max:
                diff = cols_max - cols
                b_list[b] = np.hstack((b_list[b],np.zeros((rows,diff))))
            rows,cols = b_list[b].shape
            big_list_plot.append(b_list[b])
    write_image(big_list_plot,np.float32,0,output_cd_plot,rows_max,cols_max,geo_transform,projection)    
    unsupervised_classification_otb(output_cd_plot,output_cd_plot[:-4]+'_class.TIF',n_classes,100000)
    band_list_unsup = read_image(output_cd_plot[:-4]+'_class.TIF',np.uint8,0)
    band_list_unsup[0] = band_list_unsup[0]+np.ones(band_list_unsup[0].shape)
    write_image(band_list_unsup,np.uint8,0,output_cd_plot[:-4]+'_reclass.TIF',rows_max,cols_max,geo_transform,projection)
    clip_rectangular(output_cd_plot[:-4]+'_reclass.TIF',np.uint8,input_shapefile,output_cd_plot[:-4]+'_reclass_clip.TIF',mask=True)
    
    for i in range(1,13):
        indexes_list = ["Index"+str(i)]
        sat_folder = sat_folder + separator     
        dirs = os.listdir(sat_folder)
        dirs.sort()
        dirs = [dir for dir in dirs if os.path.isdir(sat_folder+dir)]
        output_differencedays = ["Class"+str(c)+"\t" for c in range(n_classes+1)]
        output_differencedays.insert(0,"\t")
        output_differencedays = [output_differencedays]
        output_mean = ["Class"+str(c)+"\t" for c in range(n_classes+1)]
        output_mean.insert(0,"\t")
        output_mean = [output_mean]
        output_std = ["Class"+str(c)+"\t" for c in range(n_classes+1)]
        output_std.insert(0,"\t")
        output_std = [output_std]
        output_npixel = ["Class"+str(c)+"\t" for c in range(n_classes+1)]
        output_npixel.insert(0,"\t")
        output_npixel = [output_npixel]
        target_directories = list(sat_folder+directory+separator for directory in dirs if not os.path.isfile(sat_folder+directory))
        stats = ("mean","std","npixel")
        for target_index,target_dir in enumerate(target_directories):
            out = classification_statistics(sat_folder+"change_detection_all_plot_reclass_clip.TIF",target_dir+'change_detection_plot.TIF',indexes_list)
            out_mean = [str(o["mean"]) for o in out]
            out_std = [str(o["std"]) for o in out]
            out_npixel = [str(o["npixel"]) for o in out]
            out_mean.insert(0,os.path.basename(os.path.normpath(target_dir)))
            out_std.insert(0,os.path.basename(os.path.normpath(target_dir)))
            out_npixel.insert(0,os.path.basename(os.path.normpath(target_dir)))
            output_std.append(out_std)
            output_mean.append(out_mean)
            output_npixel.append(out_npixel)
            current_mean = list(out_mean)
            if target_index:
                map(lambda x: x.pop(0) if len(x) != n_classes+1 else False ,[current_mean,prev_mean])
                out_differencedays = day_difference(target_dir,target_directories[target_index-1],current_mean,prev_mean)
                out_differencedays = list(out_differencedays)
                out_differencedays.insert(0,os.path.basename(os.path.normpath(target_dir)))
                output_differencedays.append(out_differencedays)
            prev_mean = current_mean
        np.savetxt('{}/{}.txt'.format(sat_folder,"Index_"+"mean"+str(i)),np.array(output_mean, dtype=np.str_),fmt="%s",delimiter='\t',)
        np.savetxt('{}/{}.txt'.format(sat_folder,"Index_"+"std"+str(i)),np.array(output_std, dtype=np.str_),fmt="%s",delimiter='\t',)
        np.savetxt('{}/{}.txt'.format(sat_folder,"Index_"+"npixel"+str(i)),np.array(output_npixel, dtype=np.str_),fmt="%s",delimiter='\t',)
        np.savetxt('{}/{}.txt'.format(sat_folder,"Index_"+"diffence"+str(i)),np.array(output_differencedays, dtype=np.str_),fmt="%s",delimiter='\t',)

import matplotlib.pyplot as plt

def graph(sat_folder, indexes_list, graph_file):
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
        output.append(classification_statistics(sat_folder+"change_detection_all_plot_reclass_clip.TIF",target_dir+'change_detection_plot.TIF',indexes_list))
    plt.plot(output)
    leg_list = [p for p in plt.plot(output)]
    leg_labels = ['Class ' + str(n+1) for n in range(0,len(leg_list))]
    plt.legend(leg_list, leg_labels,bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig(graph_file)
    plt.show()

import datetime

def day_difference(current_data,prev_data,current_mean,prev_mean):
    day_diffence = (datetime.datetime(*map(int,os.path.basename(os.path.normpath(current_data)).split("-")))-datetime.datetime(*map(int,os.path.basename(os.path.normpath(prev_data)).split("-")))).days
    current_mean = np.array(current_mean,dtype=np.float32)
    prev_mean = np.array(prev_mean,dtype=np.float32)
    return (current_mean-prev_mean)/day_diffence

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

    rows_max = max(rows,rows_class)
    cols_max = max(cols,cols_class)
    index = int(input_index[0].replace("Index",""))
    if rows < rows_max:
        diff = rows_max - rows
        band_list[index-1] = np.vstack((band_list[index-1],np.zeros((diff,cols))))
    elif rows_class < rows_max:
        diff = rows_max - rows_class
        band_list_classification[0] = np.vstack((band_list_classification[0],np.zeros((diff,cols_class))))
        
    rows,cols = band_list[index-1].shape
    rows_class,cols_class = band_list_classification[0].shape

    if cols < cols_max:
        diff = cols_max - cols
        band_list[index-1] = np.hstack((band_list[index-1],np.zeros((rows,diff))))
    elif cols_class < cols_max:
        diff = cols_max - cols_class
        band_list_classification[0] = np.hstack((band_list_classification[0],np.zeros((rows_class,diff))))

    np.uint8(band_list_classification[0])
    max_class = int(np.max(band_list_classification[0]))
    output = []

    for value in range(max_class+1):
        mask = np.equal(band_list_classification[0],value)
        data = np.extract(mask,band_list[index-1])
        data_flat = data.flatten()
        mean_value = np.mean(data_flat)
        std_value = np.std(data_flat)
        npixel_value = np.size(data_flat)
        diction = {
            "mean": mean_value,
            "std": std_value,
            "npixel": npixel_value
            }
        output.append(diction)
    return output
        
if __name__ == "__main__":
    main()
