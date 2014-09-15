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
from numpy.fft import fft2, ifft2, fftshift
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
    #warnings.filterwarnings("ignore")
    arg = args()
    reference_folder = str(arg.reference_path)
    target_folder = str(arg.target_folder)
    enable_clip = (True if arg.enable_clip else None)
    if enable_clip:
        input_shape = str(arg.enable_clip[0])
    else:
        input_shape = ""
    enable_grid = (True if arg.enable_grid else None)
    if enable_grid: 
        tiling_row_factor = int(arg.enable_grid[0])
        tiling_col_factor = int(arg.enable_grid[1])
    else:
        tiling_row_factor = tiling_col_factor = 0
    enable_resampling = bool(arg.enable_resampling)
    enable_SURF = bool(arg.enable_SURF)
    enable_FFT = bool(arg.enable_FFT)
    coregistration(reference_folder,target_folder,enable_clip,input_shape,enable_grid,tiling_row_factor,tiling_col_factor,enable_resampling,enable_SURF,enable_FFT)


def args():
    parser = argparse.ArgumentParser(description='Calculate Features')
    parser.add_argument("reference_path", help="????")
    parser.add_argument("target_folder", help="????")
    parser.add_argument("--enable_clip", nargs=1, help="????")
    parser.add_argument("--enable_grid", nargs=2, help="????")
    parser.add_argument("--enable_resampling", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--enable_SURF", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--enable_FFT", default=False, const=True, nargs='?', help="????")
    args = parser.parse_args()
    return args


def EUC_SURF(ref_band_mat,target_band_mat,output_as_array):
     
    '''
    SURF version used for Landsat by EUCENTRE
    
    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :param output_as_array: if True the output is converted to matrix for visualization purposes
    :returns: points from reference, points from target, result of matching function or array of points (depending on the output_as_array flag)
    
    '''
    detector = cv2.FeatureDetector_create("SURF") #Detector definition
    descriptor = cv2.DescriptorExtractor_create("BRIEF") #Descriptor definition
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming") #Matcher definition
    
    #Extraction of features from REFERENCE
    ref_mask_zeros = np.ma.masked_equal(ref_band_mat, 0).astype('uint8')
    k_ref = detector.detect(ref_band_mat.astype(np.uint8), mask=ref_mask_zeros)
    kp_ref, d_ref = descriptor.compute(ref_band_mat, k_ref)
    h_ref, w_ref = ref_band_mat.shape[:2]
    ref_band_mat = []

    #Extration of features from TARGET
    target_mask_zeros = np.ma.masked_equal(target_band_mat, 0).astype('uint8')
    k_target = detector.detect(target_band_mat.astype(np.uint8), mask=target_mask_zeros)
    kp_target, d_target = descriptor.compute(target_band_mat, k_target)
    h_target, w_target = target_band_mat.shape[:2]
    target_band_mat = []

    #Matching
    matches = matcher.match(d_ref, d_target)
    matches = sorted(matches, key = lambda x:x.distance)
    matches_disp = matches[:3]
    if output_as_array == True:
        ext_points = np.zeros(shape=(len(matches_disp),4))
        i = 0
        for m in matches_disp:
            ext_points[i][:]= [int(kp_ref[m.queryIdx].pt[0]),int(kp_ref[m.queryIdx].pt[1]),int(kp_target[m.trainIdx].pt[0]),int(kp_target[m.trainIdx].pt[1])]
            i = i+1
        return kp_ref,kp_target,ext_points
    else:
        return kp_ref,kp_target,matches


def FFT_coregistration(ref_band_mat,target_band_mat):

    '''
    Alternative method used to coregister the images based on the FFT

    :param ref_band_mat: numpy 8 bit array containing reference image
    :param target_band_mat: numpy 8 bit array containing target image
    :returns: the shift among the two input images 

    '''
    status = Bar(3, "FFT")
    #Normalization - http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation 
    ref_band_mat = (ref_band_mat - ref_band_mat.mean()) / ref_band_mat.std()
    target_band_mat = (target_band_mat - target_band_mat.mean()) / target_band_mat.std() 

    #Check dimensions - they have to match
    rows_ref,cols_ref =  ref_band_mat.shape
    rows_target,cols_target = target_band_mat.shape

    if rows_target < rows_ref:
        print 'Rows - correction needed'

        diff = rows_ref - rows_target
        target_band_mat = np.vstack((target_band_mat,np.zeros((diff,cols_target))))
    elif rows_ref < rows_target:
        print 'Rows - correction needed'
        diff = rows_target - rows_ref
        ref_band_mat = np.vstack((ref_band_mat,np.zeros((diff,cols_ref))))
    status(1)
    rows_target,cols_target = target_band_mat.shape
    rows_ref,cols_ref = ref_band_mat.shape

    if cols_target < cols_ref:
        print 'Columns - correction needed'
        diff = cols_ref - cols_target
        target_band_mat = np.hstack((target_band_mat,np.zeros((rows_target,diff))))
    elif cols_ref < cols_target:
        print 'Columns - correction needed'
        diff = cols_target - cols_ref
        ref_band_mat = np.hstack((ref_band_mat,np.zeros((rows_ref,diff))))

    rows_target,cols_target = target_band_mat.shape   
    status(2)
    #translation(im_target,im_ref)
    freq_target = fft2(target_band_mat)   
    freq_ref = fft2(ref_band_mat)  
    inverse = abs(ifft2((freq_target * freq_ref.conjugate()) / (abs(freq_target) * abs(freq_ref))))   
    #Converts a flat index or array of flat indices into a tuple of coordinate arrays. would give the pixel of the max inverse value
    y_shift,x_shift = np.unravel_index(np.argmax(inverse),(rows_target,cols_target))

    if y_shift > rows_target // 2: # // used to truncate the division
        y_shift -= rows_target
    if x_shift > cols_target // 2: # // used to truncate the division
        x_shift -= cols_target
    status(3)
    return -x_shift, -y_shift


def extract_tiles(input_raster,start_col_coord,start_row_coord,end_col_coord,end_row_coord):
    
    '''
    Extract a subset of a raster according to the desired coordinates

    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param start_col_coord: starting longitude coordinate
    :param start_row_coord: starting latitude coordinate
    :param end_col_coord: ending longitude coordinate
    :param end_row_coord: ending latitude coordinate

    :returns: an output file is created and also a level of confidence on the tile is returned

    Author: Daniele De Vecchi
    Last modified: 20/08/2014
    '''

    #Read input image
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)
    band_list = read_image(input_raster,np.uint8,0)
    #Definition of the indices used to tile
    start_col_ind,start_row_ind = world2pixel(geotransform,start_col_coord,start_row_coord)
    end_col_ind,end_row_ind = world2pixel(geotransform,end_col_coord,end_row_coord)
    #print start_col_ind,start_row_ind
    #print end_col_ind,end_row_ind
    #New geotransform matrix
    new_geotransform = [start_col_coord,geotransform[1],0.0,start_row_coord,0.0,geotransform[5]]
    #Extraction
    data = band_list[0][start_row_ind:end_row_ind,start_col_ind:end_col_ind]
    
    band_list = []
    return data,start_col_coord,start_row_coord,end_col_coord,end_row_coord


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


def slope_filter(ext_points):

    '''
    Filter based on the deviation of the slope

    :param ext_points: array with coordinates of extracted points
    :returns: an array of filtered points

    Author: Daniele De Vecchi
    Last modified: 19/08/2014
    '''
    
    discard_list = []
    for p in range(0,len(ext_points)):
        #The first point is the one with minimum distance so it is supposed to be for sure correct
        x_ref,y_ref,x_target,y_target = int(ext_points[p][0]),int(ext_points[p][1]),int(ext_points[p][2]),int(ext_points[p][3])
        if x_target-x_ref != 0:
            istant_slope = float((y_target-y_ref)) / float((x_target-x_ref))
        else:
            istant_slope = 0
        if p == 0:
            slope_mean = istant_slope
        else:
            slope_mean = float(slope_mean+istant_slope) / float(2)
        slope_std = istant_slope - slope_mean
        if abs(slope_std) >= 0.1:
            discard_list.append(p)
        #print 'istant_slope: ' + str(istant_slope)
        #print 'slope_mean: ' + str(slope_mean)
        #print 'slope_std: ' + str(slope_std)
        
    new_points = np.zeros(shape=(len(ext_points)-len(discard_list),4))
    p = 0
    if len(new_points)>1:
        for dp in range(0,len(ext_points)):
            if dp not in discard_list:
                new_points[p][:]= int(ext_points[dp][0]),int(ext_points[dp][1]),int(ext_points[dp][2]),int(ext_points[dp][3])
                p = p+1
            else:
                dp = dp+1
    return new_points


def coregistration(reference_folder,target_folder,enable_clip,input_shape,enable_grid,tiling_row_factor,tiling_col_factor,enable_resampling,enable_SURF,enable_FFT):

    '''
    Function for the script

    BAND1 is used for reference and target
    '''
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'

    ref_images = os.listdir(reference_folder)
    reference_file = [s for s in ref_images if "B1.TIF" in s or "B1.tif" in s and "aux.xml" not in s]
    image_ref = reference_folder + separator + reference_file[0]
    rows_ref_orig,cols_ref_orig,nbands_ref_orig,geotransform_ref_orig,projection_ref_orig = read_image_parameters(image_ref)
    ref_new_resolution = geotransform_ref_orig[1] / float(2) 

    target_images = os.listdir(target_folder)
    print target_images
    target_file = [s for s in target_images if "B1.TIF" in s or "B1.tif" in s and "aux.xml" not in s]
    print "\n\n\n\n\n"
    print  target_file[0]
    print "PROVA"
    image_target = target_folder + separator + target_file[0]
    rows_target_orig,cols_target_orig,nbands_target_orig,geotransform_target_orig,projection_target_orig = read_image_parameters(image_target)
    target_new_resolution = geotransform_target_orig[1] / float(2)

    if enable_clip == True:
        status = Bar(100, "Clipping")
        status(0)
        clip_rectangular(image_ref,np.uint8,input_shape,image_ref[:-4]+'_roi.tif')
        image_ref = image_ref[:-4]+'_roi.tif'
        if enable_resampling == True:
            resampling(image_ref,image_ref[:-4]+'_rs_'+str(ref_new_resolution)+'.tif',ref_new_resolution,'bicubic')
        clip_rectangular(image_target,np.uint8,input_shape,image_target[:-4]+'_roi.tif')
        image_target = image_target[:-4]+'_roi.tif'
        if enable_resampling == True:
            resampling(image_target,image_target[:-4]+'_rs_'+str(target_new_resolution)+'.tif',target_new_resolution,'bicubic')
        status(100)
    elif enable_grid == True:
        minx_ref,miny_ref,maxx_ref,maxy_ref = get_coordinate_limit(image_ref)
        minx_target,miny_target,maxx_target,maxy_target = get_coordinate_limit(image_target)
        
        minx = np.max([minx_target,minx_ref])
        miny = np.max([miny_target,miny_ref])
        maxx = np.min([maxx_target,maxx_ref])
        maxy = np.min([maxy_target,maxy_ref])

        status = Bar(tiling_row_factor, "Clipping")
        for t_row in range(0,tiling_row_factor):
            for t_col in range(0,tiling_col_factor):
                start_col_coord = float(maxx-minx)*(float(t_col)/float(tiling_col_factor)) + float(minx)
                start_row_coord = float(miny-maxy)*(float(t_row)/float(tiling_row_factor)) + float(maxy)
                end_col_coord = float(maxx-minx)*(float(t_col+1)/float(tiling_col_factor)) + float(minx)
                if end_col_coord > maxx: end_col_coord = maxx
                end_row_coord = float(miny-maxy)*(float(t_row+1)/float(tiling_row_factor)) + float(maxy)
                if end_row_coord < miny: end_row_coord = miny

                band_mat,start_col_coord,start_row_coord,end_col_coord,end_row_coord = extract_tiles(image_target,start_col_coord,start_row_coord,end_col_coord,end_row_coord)
                new_geotransform_target = [start_col_coord,geotransform_target_orig[1],0.0,start_row_coord,0.0,geotransform_target_orig[5]]
                rows_target,cols_target = band_mat.shape

                target_tiles_prop = tile_statistics(band_mat,start_col_coord,start_row_coord,end_col_coord,end_row_coord)
                target_tiles_prop_list.append(target_tiles_prop)
            status(t_row)

        band_mat = None
        target_tiles_prop_list = sorted(target_tiles_prop_list,key=itemgetter(4),reverse=True)
        target_tiles_prop_list_sorted = sorted(target_tiles_prop_list,key=itemgetter(7))
        target_tiles_prop_list_sorted = [element for element in target_tiles_prop_list_sorted if element[4] != 0 and element[4] != 0.5]      

        status = Bar(2, "Clipping")
        for et in range(0,2):
            status(et)
            band_mat_target,start_col_coord_target,start_row_coord_target,end_col_coord_target,end_row_coord_target = extract_tiles(image_target,target_tiles_prop_list_sorted[et][0],target_tiles_prop_list_sorted[et][1],target_tiles_prop_list_sorted[et][2],target_tiles_prop_list_sorted[et][3])
            band_mat_ref,start_col_coord_ref,start_row_coord_ref,end_col_coord_ref,end_row_coord_ref = extract_tiles(image_ref,target_tiles_prop_list_sorted[et][0],target_tiles_prop_list_sorted[et][1],target_tiles_prop_list_sorted[et][2],target_tiles_prop_list_sorted[et][3])

            new_geotransform_target = [start_col_coord_target,geotransform_target_orig[1],0.0,start_row_coord_target,0.0,geotransform_target_orig[5]]
            new_geotransform_ref = [start_col_coord_ref,geotransform_ref_orig[1],0.0,start_row_coord_ref,0.0,geotransform_ref_orig[5]]

            rows_target,cols_target = band_mat_target.shape
            rows_ref,cols_ref = band_mat_ref.shape

            write_image([band_mat_target],np.uint8,0,image_target[:-4]+'_temp_tile_'+str(et)+'.tif',rows_target,cols_target,new_geotransform_target,projection_target_orig)
            if enable_resampling == True:
                resampling(image_target[:-4]+'_temp_tile_'+str(et)+'.tif',image_target[:-4]+'_temp_tile_'+str(et)+'_rs_'+str(target_new_resolution)+'.tif',target_new_resolution,'bicubic')
                target_tiles_list.append(image_target[:-4]+'_temp_tile_'+str(et)+'_rs_'+str(target_new_resolution)+'.tif')
            else:
                target_tiles_list.append(image_target[:-4]+'_temp_tile_'+str(et)+'.tif')

            write_image([band_mat_ref],np.uint8,0,image_ref[:-4]+'_temp_tile_'+str(et)+'.tif',rows_ref,cols_ref,new_geotransform_ref,projection_ref_orig)
            if enable_resampling == True:
                resampling(image_ref[:-4]+'_temp_tile_'+str(et)+'.tif',image_ref[:-4]+'_temp_tile_'+str(et)+'_rs_'+str(ref_new_resolution)+'.tif',ref_new_resolution,'bicubic')
                ref_tiles_list.append(image_ref[:-4]+'_temp_tile_'+str(et)+'_rs_'+str(ref_new_resolution)+'.tif')
            else:
                ref_tiles_list.append(image_ref[:-4]+'_temp_tile_'+str(et)+'.tif')

    if enable_SURF == True:

        if enable_clip == True:
            band_list_ref = read_image(image_ref,np.uint8,0)
            rows_ref,cols_ref,nbands_ref,geotransform_ref,projection_ref = read_image_parameters(image_ref)
            band_list_target = read_image(image_target,np.uint8,0)
            rows_target,cols_target,nbands_target,geotransform_target,projection_target = read_image_parameters(image_target)

            kp_ref,kp_target,ext_points = EUC_SURF(band_list_ref[0],band_list_target[0],output_as_array=True)
            ext_points = slope_filter(ext_points)
            status = Bar(len(ext_points), "SURF")
            x_shift_surf, y_shift_surf = 0,0
            if len(ext_points) > 1:
                for p in range(0,len(ext_points)):
                    status(p)
                    x_ref,y_ref,x_target,y_target = int(ext_points[p][0]),int(ext_points[p][1]),int(ext_points[p][2]),int(ext_points[p][3])
                    x_shift_surf = x_shift_surf + (x_ref-x_target)
                    y_shift_surf = y_shift_surf + (y_ref-y_target)

                x_shift_surf = float(x_shift_surf) / float(len(ext_points))
                y_shift_surf = float(y_shift_surf) / float(len(ext_points))

            #M = np.float32([[1,0,x_shift_surf],[0,1,y_shift_surf]])
            band_list_ref = []
            band_list_target = []

        elif enable_grid == True:
            ext_points_list = []
            status = Bar(target_tiles_list, "SURF 1/2")
            for f in range(0,len(target_tiles_list)):
                status(f)
                band_list_target = read_image(target_tiles_list[f],np.uint8,0)
                band_list_ref = read_image(ref_tiles_list[f],np.uint8,0)

                kp_ref,kp_target,ext_points = EUC_SURF(band_list_ref[0],band_list_target[0],output_as_array=True)
                ext_points = slope_filter(ext_points)
                ext_points_list.append(ext_points)
                
                band_list_target = []
                band_list_ref = []    

            x_shift_surf, y_shift_surf = 0,0
            status = Bar(ext_points_list, "SURF 2/2")
            for f in range(0,len(ext_points_list)):
                status(f)
                for p in range(0,len(ext_points_list[f])):
                    x_ref,y_ref,x_target,y_target = int(ext_points_list[f][p][0]),int(ext_points_list[f][p][1]),int(ext_points_list[f][p][2]),int(ext_points_list[f][p][3])
                    x_shift_surf = x_shift_surf + (x_ref-x_target)
                    y_shift_surf = y_shift_surf + (y_ref-y_target)

            
            x_shift_surf = float(x_shift_surf) / float(len(ext_points_list[0])+len(ext_points_list[1]))
            y_shift_surf = float(y_shift) / float(len(ext_points_list[0])+len(ext_points_list[1]))

        M_surf = np.float32([[1,0,x_shift_surf],[0,1,y_shift_surf]])

    if enable_FFT == True:

        if enable_clip == True:
            band_list_ref = read_image(image_ref,np.uint8,0)
            band_list_target = read_image(image_target,np.uint8,0)
            x_shift_fft,y_shift_fft = FFT_coregistration(band_list_ref[0],band_list_target[0])

            band_list_ref = []
            band_list_target = []

        elif enable_grid == True:
            x_shift_fft_tot = 0
            y_shift_fft_tot = 0
            for f in range(0,len(target_tiles_list)):
                band_list_target = read_image(target_tiles_list[f],np.uint8,0)
                band_list_ref = read_image(ref_tiles_list[f],np.uint8,0)
                x_shift_fft,y_shift_fft = FFT_coregistration(band_list_ref[0],band_list_target[0])
                x_shift_fft_tot = x_shift_fft_tot + x_shift_fft
                y_shift_fft_tot = y_shift_fft_tot + y_shift_fft

                band_list_ref = []
                band_list_target = []

            x_shift_fft = float(x_shift_fft_tot) / float(len(target_tiles_list))
            y_shift_fft = float(y_shift_fft_tot) / float(len(target_tiles_list))

        M_fft = np.float32([[1,0,x_shift_fft],[0,1,y_shift_fft]])

    target_rs = [s for s in target_images if ".TIF" in s or ".tif" in s and "aux.xml" not in s]
    for tg_file in target_rs:
        tg_file = target_folder + separator + tg_file
        if enable_resampling == True:
            resampling(tg_file,tg_file[:-4]+'_rs_'+str(target_new_resolution)+'.tif',target_new_resolution,'bicubic')
            tg_file = tg_file[:-4]+'_rs_'+str(target_new_resolution)+'.tif'
            band_list_target = read_image(tg_file,np.uint8,0)
            rows_target,cols_target,nbands_target,geotransform_target,projection = read_image_parameters(tg_file)
        else:
            band_list_target = read_image(tg_file,np.uint8,0)
            rows_target,cols_target,nbands_target,geotransform_target,projection = read_image_parameters(tg_file)
            
        if enable_FFT == True:
            dst = cv2.warpAffine(band_list_target[0],M_fft,(cols_target,rows_target))
            write_image([dst],np.uint8,0,tg_file[:-4]+'_adj_fft.tif',rows_target,cols_target,geotransform_target,projection_target)

            if enable_resampling == True:
                target_original_resolution = geotransform_target_orig[1]
                resampling(tg_file[:-4]+'_adj_fft.tif',tg_file[:-4]+'adj_fft_rs_'+str(target_original_resolution)+'.tif',target_original_resolution,'bicubic')
            
        if enable_SURF == True:
            dst = cv2.warpAffine(band_list_target[0],M_surf,(cols_target,rows_target))
            write_image([dst],np.uint8,0,tg_file[:-4]+'_adj_surf.tif',rows_target,cols_target,geotransform_target,projection_target)

            if enable_resampling == True:
                target_original_resolution = geotransform_target_orig[1]
                resampling(tg_file[:-4]+'_adj_surf.tif',tg_file[:-4]+'_adj_surf_rs_'+str(target_original_resolution)+'.tif',target_original_resolution,'bicubic')

if __name__ == "__main__":
    main()
    