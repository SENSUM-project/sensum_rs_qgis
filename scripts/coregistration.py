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
    enable_clip = (True if arg.enable_clip else False)
    if enable_clip == True:
        input_shape = str(arg.enable_clip[0])
    else:
        input_shape = ""
    enable_grid = (True if arg.enable_grid else None)
    if enable_grid: 
        tiling_row_factor = int(arg.enable_grid[0])
        tiling_col_factor = int(arg.enable_grid[1])
        print tiling_row_factor,tiling_col_factor
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
    target_file = [s for s in target_images if "B1.TIF" in s or "B1.tif" in s and "aux.xml" not in s]
    print "\n\n\n\n\n"
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
        target_tiles_prop_list = []
        status = Bar(tiling_row_factor, "Tile definition")
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

        status = Bar(2, "Tile extraction")
        target_tiles_list = []
        ref_tiles_list = []
        for et in range(0,2):
            status(et)
            rows_target,cols_target,nbands_target,geotransform_target,projection_target = read_image_parameters(image_target)
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

            #kp_ref,kp_target,ext_points = points_extraction(band_list_ref[0],band_list_target[0],output_as_array=True)
            ext_point = points_extraction(band_list_ref[0],band_list_target[0],output_as_array=True)
            #print ext_point
            x_shift_surf = ext_point[0][0] - ext_point[0][2]
            y_shift_surf = ext_point[0][1] - ext_point[0][3]
            

            band_list_ref = []
            band_list_target = []
            os.remove(image_ref)
            os.remove(image_target)

        elif enable_grid == True:
            x_shift_surf_tot = 0
            y_shift_surf_tot = 0
            #status = Bar(target_tiles_list, "SURF 1/2")
            for f in range(0,len(target_tiles_list)):
                status(f)
                band_list_target = read_image(target_tiles_list[f],np.uint8,0)
                band_list_ref = read_image(ref_tiles_list[f],np.uint8,0)

                #kp_ref,kp_target,ext_points = points_extraction(band_list_ref[0],band_list_target[0],output_as_array=True)
                #ext_points = slope_filter(ext_points)
                ext_point = points_extraction(band_list_ref[0],band_list_target[0],output_as_array=True)
                
                band_list_target = []
                band_list_ref = []   
                os.remove(target_tiles_list[f]) 
                os.remove(ref_tiles_list[f])
                x_shift_surf_tot = x_shift_surf_tot + x_shift_surf
                y_shift_surf_tot = y_shift_surf_tot + y_shift_surf

            x_shift_surf = float(x_shift_surf_tot) / float(len(target_tiles_list))
            y_shift_surf = float(y_shift_surf_tot) / float(len(target_tiles_list))

        if (x_shift_surf == 1 and y_shift_surf == 0) or (x_shift_surf == 0 and y_shift_surf == 1):
                x_shift_surf,y_shift_surf = [0, 0]
        M_surf = np.float32([[1,0,x_shift_surf],[0,1,y_shift_surf]])
        print M_surf

    if enable_FFT == True:

        if enable_clip == True:
            band_list_ref = read_image(image_ref,np.uint8,0)
            band_list_target = read_image(image_target,np.uint8,0)
            x_shift_fft,y_shift_fft = FFT_coregistration(band_list_ref[0],band_list_target[0])
            print 'FFT Shift -> X:' + str(x_shift_fft) +' Y: ' + str(y_shift_fft)

            band_list_ref = []
            band_list_target = []

        elif enable_grid == True:
            x_shift_fft_tot = 0
            y_shift_fft_tot = 0
            for f in range(0,len(target_tiles_list)):
                band_list_target = read_image(target_tiles_list[f],np.uint8,0)
                band_list_ref = read_image(ref_tiles_list[f],np.uint8,0)
                x_shift_fft,y_shift_fft = FFT_coregistration(band_list_ref[0],band_list_target[0])
                print 'FFT Shift -> X:' + str(x_shift_fft) +' Y: ' + str(y_shift_fft)
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
            rows_target,cols_target,nbands_target,geotransform_target,projection_target = read_image_parameters(tg_file)
            
        if enable_FFT == True:
            shutil.copyfile(tg_file,tg_file[:-4]+'_adj_surf.tif')
            rows_target,cols_target,nbands_target,geotransform_target,projection = read_image_parameters(tg_file)
            new_lon = float(x_shift_fft*geotransform_target[1]+geotransform_target[0]) 
            new_lat = float(geotransform_target[3]+y_shift_fft*geotransform_target[5])
            fixed_geotransform = [new_lon,geotransform_target[1],0.0,new_lat,0.0,geotransform_target[5]]
    
            up_image = osgeo.gdal.Open(tg_file[:-4]+'_adj_surf.tif', GA_Update)
            up_image.SetGeoTransform(fixed_geotransform)
            up_image = None

            if enable_resampling == True:
                target_original_resolution = geotransform_target_orig[1]
                resampling(tg_file[:-4]+'_adj_fft.tif',tg_file[:-4]+'adj_fft_rs_'+str(target_original_resolution)+'.tif',target_original_resolution,'bicubic')
           
        if enable_SURF == True:
            shutil.copyfile(tg_file,tg_file[:-4]+'_adj_surf.tif')
            rows_target,cols_target,nbands_target,geotransform_target,projection = read_image_parameters(tg_file)
            new_lon = float(x_shift_surf*geotransform_target[1]+geotransform_target[0]) 
            new_lat = float(geotransform_target[3]+y_shift_surf*geotransform_target[5])
            fixed_geotransform = [new_lon,geotransform_target[1],0.0,new_lat,0.0,geotransform_target[5]]
    
            up_image = osgeo.gdal.Open(tg_file[:-4]+'_adj_surf.tif', GA_Update)
            up_image.SetGeoTransform(fixed_geotransform)
            up_image = None

            if enable_resampling == True:
                target_original_resolution = geotransform_target_orig[1]
                resampling(tg_file[:-4]+'_adj_surf.tif',tg_file[:-4]+'_adj_surf_rs_'+str(target_original_resolution)+'.tif',target_original_resolution,'bicubic')
        
if __name__ == "__main__":
    main()
    