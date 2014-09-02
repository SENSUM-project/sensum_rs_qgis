#!/usr/bin/python
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
import subprocess
from utils import Bar

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import sensum_library.config
from sensum_library.preprocess import *
from sensum_library.classification import *
from sensum_library.segmentation import *
from sensum_library.conversion import *
from sensum_library.segmentation_opt import *
from sensum_library.features import *
from sensum_library.secondary_indicators import *

def main():
    multiband_pre = "/home/gale/Test_HR_change_detection/Muzaffarabad_QB_pre_post_R1_RAW/052006380010_01-20040813/052006380010_01_P001_MUL/04AUG13055047-M2AS-052006380010_01_P001.TIF"
    panchromatic_pre = "/home/gale/Test_HR_change_detection/Muzaffarabad_QB_pre_post_R1_RAW/052006380010_01-20040813/052006380010_01_P001_PAN/04AUG13055047-P2AS-052006380010_01_P001.TIF"
    multiband_post = "/home/gale/Test_HR_change_detection/Muzaffarabad_QB_pre_post_R1_RAW/052006380030_01-20051022/052006380030_01_P001_MUL/05OCT22060051-M2AS-052006380030_01_P001.TIF"
    panchromatic_post = "/home/gale/Test_HR_change_detection/Muzaffarabad_QB_pre_post_R1_RAW/052006380030_01-20051022/052006380030_01_P001_PAN/05OCT22060051-P2AS-052006380030_01_P001.TIF"
    clip_shape = "/home/gale/Test_HR_change_detection/Building_set1.shp"
    change_detection(multiband_pre,panchromatic_pre,multiband_post,panchromatic_post,clip_shape)

def change_detection(multiband_pre,panchromatic_pre,multiband_post,panchromatic_post,clip_shape):
    pansharp_pre = os.path.splitext(multiband_pre)[0]+'_pansharp.tif'
    pansharp_post = os.path.splitext(multiband_post)[0]+'_pansharp.tif'
    pansharp_pre_clipped = os.path.splitext(multiband_pre)[0]+'_pansharp_clipped.tif'
    pansharp_post_clipped = os.path.splitext(multiband_post)[0]+'_pansharp_clipped.tif'
    panchromatic_pre_clipped = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_clipped.tif'
    panchromatic_post_clipped = os.path.splitext(panchromatic_post)[0]+'_panchromatic_clipped.tif'
    panchromatic_roughness_pre = os.path.splitext(multiband_pre)[0]+'_roughness.tif'
    panchromatic_roughness_post = os.path.splitext(multiband_post)[0]+'_roughness.tif'
    pansharp_roughness_pre = os.path.splitext(pansharp_pre)[0]+'_roughness.tif'
    pansharp_roughness_post = os.path.splitext(pansharp_post)[0]+'_roughness.tif'
    panchromatic_pre_buffer = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_buffer.tif'
    panchromatic_post_buffer = os.path.splitext(panchromatic_post)[0]+'_panchromatic_buffer.tif'
    panchromatic_pre_canny = os.path.splitext(panchromatic_pre)[0]+'_canny.tif'
    panchromatic_post_canny = os.path.splitext(panchromatic_post)[0]+'_canny.tif'
    panchromatic_pre_clipped_slope = os.path.splitext(panchromatic_pre)[0]+'_slope.tif'
    panchromatic_post_clipped_slope = os.path.splitext(panchromatic_post)[0]+'_slope.tif'
    '''
    #1
    pansharp(multiband_pre, panchromatic_pre)
    pansharp(multiband_post, panchromatic_post)
    #2
    ####CO-REGISTRATION####
    #3
    clip_rectangular(pansharp_pre,0,clip_shape,pansharp_pre_clipped)
    clip_rectangular(pansharp_post,0,clip_shape,pansharp_post_clipped)
    clip_rectangular(panchromatic_pre,0,clip_shape,panchromatic_pre_clipped)
    clip_rectangular(panchromatic_post,0,clip_shape,panchromatic_post_clipped)
    #4
    '''
    ncdi_panchromatic_clipped = NCDI(panchromatic_pre_clipped,panchromatic_post_clipped)
    ncdi_pansharp_clipped = NCDI(pansharp_pre_clipped,pansharp_post_clipped)
    '''
    roughness(panchromatic_pre_clipped, panchromatic_roughness_pre)
    roughness(panchromatic_post_clipped, panchromatic_roughness_post)
    roughness(pansharp_pre_clipped, pansharp_roughness_pre)
    roughness(pansharp_post_clipped, pansharp_post_clipped)
    #5
    pansharp_pre_stats = zonal_stats(clip_shape,pansharp_pre_clipped)
    pansharp_post_stats = zonal_stats(clip_shape,pansharp_post_clipped)
    panchromatic_pre_stats_clipped = zonal_stats(clip_shape,panchromatic_pre_clipped)
    panchromatic_post_stats_clipped = zonal_stats(clip_shape,panchromatic_post_clipped)
    #6
    buffer_2meters(clip_shape, panchromatic_pre, panchromatic_pre_buffer)
    buffer_2meters(clip_shape, panchromatic_post, panchromatic_post_buffer)
    #7
    canny(panchromatic_pre_buffer, panchromatic_pre_canny)
    canny(panchromatic_post_buffer, panchromatic_post_canny)
    #8
    ncdi_panchromatic_buffer = NCDI(panchromatic_pre_buffer,panchromatic_post_buffer)
    #9
    panchromatic_pre_stats_buffer = zonal_stats(clip_shape,panchromatic_pre_buffer)
    panchromatic_post_stats_buffer = zonal_stats(clip_shape,panchromatic_post_buffer)
    #10
    slope(panchromatic_pre_clipped, panchromatic_pre_clipped_slope)
    slope(panchromatic_post_clipped, panchromatic_post_clipped_slope)
    clip_rectangular(multiband_pre,0,clip_shape,multiband_pre_clipped)
    clip_rectangular(multiband_post,0,clip_shape,multiband_post_clipped)
    slope(multiband_pre_clipped, multiband_pre_clipped_slope)
    slope(multiband_post_clipped, multiband_post_clipped_slope)
    #11
    ncdi_multiband_clipped = NCDI(multiband_pre_clipped,multiband_post_clipped)
    #12
    multiband_pre_stats = zonal_stats(clip_shape,multiband_pre_clipped)
    multiband_post_stats = zonal_stats(clip_shape,multiband_post_clipped)
    '''

def executeCmd(command):
    os.system(command)

def executeGdal(command):
    command = (command if os.name == "posix" else "python.exe "+os.path.dirname(os.path.abspath(__file__))+command)
    print command
    executeCmd(command)

def pansharp(multiband_image, panchromatic_image):
    band_list = read_image(multiband_image,0,0)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(multiband_image)
    write_image(band_list,0,0,multiband_image,rows,cols,geo_transform,projection)
    rowsp,colsp,nbands,geo_transform,projection = read_image_parameters(panchromatic_image)
    rowsxs,colsxs,nbands,geo_transform,projection = read_image_parameters(multiband_image)
    scale_rows = round(float(rowsp)/float(rowsxs),0)
    scale_cols = round(float(colsp)/float(colsxs),0)
    executeCmd("otbcli_RigidTransformResample -progress 1 -in {} -out {} -transform.type id -transform.type.id.scalex {} -transform.type.id.scaley {}".format(multiband_image,multiband_image[:-4]+'_resampled.tif',scale_cols,scale_rows))
    fix_tiling_raster(panchromatic_image,multiband_image[:-4]+'_resampled.tif')
    executeCmd("otbcli_Pansharpening -progress 1 -inp {} -inxs {} -out {} uint16".format(panchromatic_image,multiband_image[:-4]+'_resampled.tif',multiband_image[:-4]+'_pansharp.tif'))

def NCDI(raster_pre,raster_post):
    pre = np.array(read_image(raster_pre,0,0))
    post = np.array(read_image(raster_post,0,0))
    return ((pre-post)/(pre+post)*np.sqrt(np.absolute((pre-post)/(pre+post))))/np.absolute((pre-post)/(pre+post)++0.00001)

def roughness(input_dem,output_roughness_map):
    executeGdal("gdaldem roughness {} {}".format(input_dem, output_roughness_map))

def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1
    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1
    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)

def zonal_stats(vector_path, raster_path):
    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()
    vds = ogr.Open(vector_path, GA_ReadOnly)  # TODO maybe open update if we want to write stats
    assert(vds)
    vlyr = vds.GetLayer(0)
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')
    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    while feat is not None:
        # use local source extent
        src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
        src_array = rb.ReadAsArray(*src_offset)
        # calculate new geotransform of the feature subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )
        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == None,
                np.logical_not(rv_array)
            )
        )
        feature_stats = {
            'min': float(masked.min()),
            'mean': float(masked.mean()),
            'max': float(masked.max()),
            'std': float(masked.std()),
            'sum': float(masked.sum()),
            'count': int(masked.count()),
            'fid': int(feat.GetFID())}
        stats.append(feature_stats)
        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()
    vds = None
    rds = None
    return stats

class Donut(object):

    def init(self,feature,radius=0):
        self.feature = feature

    def add(self):
        geom = self.feature.GetGeometryRef()
        donut = geom.Buffer(radius,40)
        return donut

def buffer_2meters(inputShape,panchromatic,panchromatic_output):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    inputDS = driver.Open(inputShape)
    inputLayer = inputDS.GetLayer()
    inputFeaturesCount = inputLayer.GetFeatureCount()
    outDS = driver.CreateDataSource("tmp.shp")
    outLayer = outDS.CreateLayer("Donuts", None, ogr.wkbPoint )
    for i in range(inputFeaturesCount):
        print "{} di {} features".format(i+1,inputFeaturesCount)
        inputFeature = inputLayer.GetFeature(i)
        maker = WindowsMaker(inputFeature,2)
        donutFeature = maker.make_feature(polygon=Donut)
        outLayer.CreateFeature(donutFeature)
    clip_rectangular(panchromatic,0,"tmp.shp",panchromatic_output)

import cv2
from matplotlib import pyplot as plt
def canny(input,output):
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input)
    img = cv2.imread('messi5.jpg',0)
    edges = cv2.Canny(img,rows,cols)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title(input), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title(output), plt.xticks([]), plt.yticks([])
    plt.show()

def slope(input_dem,output_slope_map):
    executeGdal("gdaldem slope {} {}".format(input_dem, output_slope_map))

if __name__ == "__main__":
    main()
