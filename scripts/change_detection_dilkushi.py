#!/usr/bin/python
import os,sys
import shutil
import time
import tempfile
import osgeo.gdal, gdal
import osgeo.ogr, ogr
from osgeo.gdalconst import *
import numpy as np
import numpy
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

RSZ = 10

def main():
    arg = args()
    multiband_pre = str(arg.multiband_pre)
    panchromatic_pre = str(arg.panchromatic_pre)
    multiband_post = str(arg.multiband_post)
    panchromatic_post = str(arg.panchromatic_post)
    clip_shape = str(arg.clip_shape)
    shp = ogr.Open(clip_shape, update=1)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    lyr.CreateField(ogr.FieldDefn("Mask", ogr.OFTInteger))
    for feature_id in featList:
        feat = lyr.GetFeature(feature_id)
        feat.SetField("Mask",1)
        lyr.SetFeature(feat)
    shp.Destroy()
    change_detection(multiband_pre,panchromatic_pre,multiband_post,panchromatic_post,clip_shape)


def args():
    parser = argparse.ArgumentParser(description='Change Detection')
    parser.add_argument("multiband_pre", help="????")
    parser.add_argument("panchromatic_pre", help="????")
    parser.add_argument("multiband_post", help="????")
    parser.add_argument("panchromatic_post", help="????")
    parser.add_argument("clip_shape", help="????")
    args = parser.parse_args()
    return args


def change_detection(multiband_pre,panchromatic_pre,multiband_post,panchromatic_post,clip_shape):
    
    pansharp_pre = os.path.splitext(multiband_pre)[0]+'_pansharp_pre.tif'
    pansharp_post = os.path.splitext(multiband_post)[0]+'_pansharp_post.tif'
    panchromatic_pre_clipped = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_pre_clipped.tif'
    panchromatic_post_clipped = os.path.splitext(panchromatic_post)[0]+'_panchromatic_post_clipped.tif'
    panchromatic_ncdi = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_ncdi.tif'
    panchromatic_ncdi_clipped = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_ncdi_clipped.tif'
    panchromatic_ncdi_roughness = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_ncdi_roughness.tif'
    panchromatic_ncdi_roughness_clipped = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_ncdi_roughness_clipped.tif'
    pansharp_pre_clipped = os.path.splitext(pansharp_pre)[0]+'_clipped.tif'
    pansharp_pre_clipped_slope = os.path.splitext(pansharp_pre)[0]+'_clipped_slope.tif'
    pansharp_post_clipped = os.path.splitext(pansharp_post)[0]+'_clipped.tif'
    pansharp_post_clipped_slope = os.path.splitext(pansharp_post)[0]+'_clipped_slope.tif'
    panchromatic_pre_clipped_slope = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_pre_slope.tif'
    panchromatic_post_clipped_slope = os.path.splitext(panchromatic_post)[0]+'_panchromatic_post_slope.tif'
    multiband_pre_clipped_slope = os.path.splitext(multiband_pre)[0]+'_multiband_pre_slope.tif'
    multiband_post_clipped_slope = os.path.splitext(multiband_post)[0]+'_multiband_post_slope.tif'
    pansharp_ncdi = os.path.splitext(pansharp_pre)[0]+'_ncdi.tif'
    pansharp_ncdi_roughness = os.path.splitext(pansharp_pre)[0]+'_ncdi_roughness.tif'
    pansharp_ncdi_roughness_clipped = os.path.splitext(pansharp_pre)[0]+'_ncdi_roughness_clipped.tif'
    multiband_pre_clipped = os.path.splitext(multiband_pre)[0]+'_multiband_pre_clipped.tif'
    multiband_post_clipped = os.path.splitext(multiband_post)[0]+'_multiband_post_clipped.tif'
    multiband_ncdi = os.path.splitext(multiband_pre)[0]+'_multiband_ncdi.tif'
    panchromatic_pre_buffer = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_pre_buffer.tif'
    panchromatic_post_buffer = os.path.splitext(panchromatic_post)[0]+'_panchromatic_post_buffer.tif'
    panchromatic_pre_canny = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_pre_canny.tif'
    panchromatic_post_canny = os.path.splitext(panchromatic_post)[0]+'_panchromatic_post_canny.tif'
    panchromatic_canny_ncdi = os.path.splitext(panchromatic_pre)[0]+'_panchromatic_canny_ncdi.tif'
    
    status = Bar(27)
    #0
    '''
    select_bands(multiband_pre,'_multiband_pre.tif')
    multiband_pre = os.path.splitext(multiband_pre)[0]+'_multiband_pre.tif'
    select_bands(multiband_post,'_multiband_post.tif')
    multiband_post = os.path.splitext(multiband_post)[0]+'_multiband_post.tif'
    status(1)
    
    #1-2
    pansharp(multiband_pre, panchromatic_pre,pansharp_pre)
    pansharp(multiband_post, panchromatic_post,pansharp_post)
    status(2)
    #3
    ####CO-REGISTRATION####
    #4-5
    clip_rectangular(panchromatic_pre,0,clip_shape,panchromatic_pre_clipped,resize=RSZ)
    clip_rectangular(panchromatic_post,0,clip_shape,panchromatic_post_clipped,resize=RSZ)
    status(5)
    
    #6
    #panchromatic NCDI
    NCDI(panchromatic_pre_clipped,panchromatic_post_clipped,panchromatic_ncdi)
    status(6)
    #7
    #panchromatic NDCI roughness
    roughness(panchromatic_ncdi, panchromatic_ncdi_roughness,'panchromatic')
    status(7)
    
    #8
    clip_rectangular(panchromatic_ncdi_roughness,0,clip_shape,panchromatic_ncdi_roughness_clipped,mask=True,resize=RSZ)
    #panchromatic NDCI roughness stats
    loop_zonal_stats(clip_shape,panchromatic_ncdi_roughness_clipped,"1")
    status(8)
    '''
    #9-10
    clip_rectangular(pansharp_pre,0,clip_shape,pansharp_pre_clipped,resize=RSZ)
    clip_rectangular(pansharp_post,0,clip_shape,pansharp_post_clipped,resize=RSZ)
    status(10)
    #11
    NCDI(pansharp_pre_clipped,pansharp_post_clipped,pansharp_ncdi)
    status(11)
    #12
    roughness(pansharp_ncdi, pansharp_ncdi_roughness)
    status(12)
    #13
    clip_rectangular(pansharp_ncdi_roughness[:-4]+'_b1.tif',0,clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b1.tif',mask=True,resize=RSZ)
    clip_rectangular(pansharp_ncdi_roughness[:-4]+'_b2.tif',0,clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b2.tif',mask=True,resize=RSZ)
    clip_rectangular(pansharp_ncdi_roughness[:-4]+'_b3.tif',0,clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b3.tif',mask=True,resize=RSZ)
    #pansharp NCDI roughness stats
    loop_zonal_stats(clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b1.tif',"b1_2")
    loop_zonal_stats(clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b2.tif',"b2_2")
    loop_zonal_stats(clip_shape,pansharp_ncdi_roughness_clipped[:-4]+'_b3.tif',"b3_2")
    status(13)
    
    #14
    buffer_2meters(clip_shape, panchromatic_pre, panchromatic_pre_buffer)
    status(14)
    #15
    buffer_2meters(clip_shape, panchromatic_post, panchromatic_post_buffer)
    status(15)
    
    #16    
    canny(panchromatic_pre_buffer, panchromatic_pre_canny)
    status(16)
    #17
    canny(panchromatic_post_buffer, panchromatic_post_canny)
    status(17)
    #18-19
    NCDI(panchromatic_pre_canny,panchromatic_post_canny,panchromatic_canny_ncdi)
    status(19)
    #20
    #panchromatic NCDI canny edge stats
    loop_zonal_stats(clip_shape,panchromatic_canny_ncdi,"3")
    status(20)
    
    #21
    slope(panchromatic_pre_clipped,panchromatic_pre_clipped_slope,'panchromatic')
    status(21)
    #22
    slope(panchromatic_post_clipped,panchromatic_post_clipped_slope,'panchromatic')
    status(22)
    #23
    NCDI(panchromatic_pre_clipped_slope,panchromatic_post_clipped_slope,panchromatic_post_clipped_slope[:-4]+'_ncdi.tif')
    #24
    #panchromatic NCDI slope stats
    loop_zonal_stats(clip_shape,panchromatic_post_clipped_slope[:-4]+'_ncdi.tif',"4")
    
    #25
    #clip_rectangular(multiband_pre,0,clip_shape,multiband_pre_clipped,resize=RSZ)
    slope(pansharp_pre_clipped, pansharp_pre_clipped_slope)
    status(23)
    #26
    slope(pansharp_post_clipped, pansharp_post_clipped_slope)
    #27
    #clip_rectangular(multiband_post,0,clip_shape,multiband_post_clipped,resize=RSZ)
    #slope(multiband_post_clipped, multiband_post_clipped_slope)
    status(24)
    #28
    NCDI(pansharp_pre_clipped_slope[:-4]+'_b1.tif',pansharp_post_clipped_slope[:-4]+'_b1.tif',pansharp_post_clipped_slope[:-4]+'_b1_ncdi.tif')
    NCDI(pansharp_pre_clipped_slope[:-4]+'_b2.tif',pansharp_post_clipped_slope[:-4]+'_b2.tif',pansharp_post_clipped_slope[:-4]+'_b2_ncdi.tif')
    NCDI(pansharp_pre_clipped_slope[:-4]+'_b3.tif',pansharp_post_clipped_slope[:-4]+'_b3.tif',pansharp_post_clipped_slope[:-4]+'_b3_ncdi.tif')
    #29 (uguale al punto 6)
    #NCDI(panchromatic_pre_clipped,panchromatic_post_clipped,panchromatic_ncdi)
    #30
    #loop_zonal_stats(clip_shape,multiband_ncdi,"4")
    loop_zonal_stats(clip_shape,pansharp_post_clipped_slope[:-4]+'_b1_ncdi.tif',"b1_5")
    loop_zonal_stats(clip_shape,pansharp_post_clipped_slope[:-4]+'_b2_ncdi.tif',"b2_5")
    loop_zonal_stats(clip_shape,pansharp_post_clipped_slope[:-4]+'_b3_ncdi.tif',"b3_5")
    status(27)
    
def executeCmd(command):
    os.system(command)


def select_bands(input_raster,ext):
    band_list = read_image(input_raster,0,0)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    if nbands > 3:
        band_list_tmp = list()
        for i in range (1,4):
            band_list_tmp.append(band_list[i])
        band_list = band_list_tmp
        del band_list_tmp
    output_raster = os.path.splitext(input_raster)[0]+ext
    write_image(band_list,0,0,output_raster,rows,cols,geo_transform,projection)


def pansharp(multiband_image, panchromatic_image,output_image):
    band_list = read_image(multiband_image,0,0)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(multiband_image)
    write_image(band_list,0,0,multiband_image,rows,cols,geo_transform,projection)
    rowsp,colsp,nbands,geo_transform,projection = read_image_parameters(panchromatic_image)
    rowsxs,colsxs,nbands,geo_transform,projection = read_image_parameters(multiband_image)
    scale_rows = round(float(rowsp)/float(rowsxs),0)
    scale_cols = round(float(colsp)/float(colsxs),0)
    executeCmd("otbcli_RigidTransformResample -progress 1 -in {} -out {} -transform.type id -transform.type.id.scalex {} -transform.type.id.scaley {}".format(multiband_image,multiband_image[:-4]+'_resampled.tif',scale_cols,scale_rows))
    fix_tiling_raster(panchromatic_image,multiband_image[:-4]+'_resampled.tif')
    executeCmd("otbcli_Pansharpening -progress 1 -inp {} -inxs {} -out {} uint16".format(panchromatic_image,multiband_image[:-4]+'_resampled.tif',output_image))

def NCDI(raster_pre,raster_post,output):
    rows,cols,nbands,geo_transform,projection = read_image_parameters(raster_pre)
    #print 'N bands: ' + str(nbands)
    if nbands == 1:
        pre = np.array(read_image(raster_pre,0,0))[0]
        post = np.array(read_image(raster_post,0,0))[0]
        ncdi = ((pre-post)/(pre+post)*np.sqrt(np.absolute((pre-post)/(pre+post))))/(np.absolute((pre-post)/(pre+post))+0.00001)
        write_image([ncdi],0,0,output,rows,cols,geo_transform,projection)
    else:
        ncdi_list = []
        for b in range(0,nbands):
            pre = np.array(read_image(raster_pre,0,0))[b]
            post = np.array(read_image(raster_post,0,0))[b]
            ncdi = ((pre-post)/(pre+post)*np.sqrt(np.absolute((pre-post)/(pre+post))))/(np.absolute((pre-post)/(pre+post))+0.00001)
            ncdi_list.append(ncdi)
        write_image(ncdi_list,0,0,output,rows,cols,geo_transform,projection)

def roughness(input_dem,output_roughness_map,input_type='pansharp'):
    if input_type == 'panchromatic':
        executeCmd("gdaldem roughness {} {}".format(input_dem, output_roughness_map))
    else:
        band_list = read_image(input_dem,np.uint16,0)
        rows_ps,cols_ps,nbands_ps,geot_ps,proj_ps = read_image_parameters(input_dem)

        write_image([band_list[0]],0,0,input_dem[:-4]+'_b1.tif',rows_ps,cols_ps,geot_ps,proj_ps)
        write_image([band_list[1]],0,0,input_dem[:-4]+'_b2.tif',rows_ps,cols_ps,geot_ps,proj_ps)
        write_image([band_list[2]],0,0,input_dem[:-4]+'_b3.tif',rows_ps,cols_ps,geot_ps,proj_ps)

        executeCmd("gdaldem roughness {} {}".format(input_dem[:-4]+'_b1.tif', output_roughness_map[:-4]+'_b1.tif'))
        executeCmd("gdaldem roughness {} {}".format(input_dem[:-4]+'_b2.tif', output_roughness_map[:-4]+'_b2.tif'))
        executeCmd("gdaldem roughness {} {}".format(input_dem[:-4]+'_b3.tif', output_roughness_map[:-4]+'_b3.tif'))


def zonal_stats(feat, input_zone_polygon, input_value_raster):
    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    nbands = raster.RasterCount
    #print 'N bands: ' + str(nbands)
    # Get extent of feat
    geom = feat.GetGeometryRef()
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)    
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)    
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
    else:
        sys.exit()
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)
    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1
    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))
    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
    # Read raster as arrays
    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(numpy.uint16)
    
    if nbands == 1:
        banddataraster = raster.GetRasterBand(1)
        dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint16)

    # Mask zone of raster
    #masked = numpy.ma.masked_array(dataraster,  numpy.logical_not(datamask))
        masked = numpy.extract(numpy.logical_not(datamask),dataraster)
        if masked.size !=0:
        # Calculate statistics of zonal raster
            feature_stats = {
                'min': masked.min(),
                'max': masked.max(),
                'sum': masked.sum(),
                'count': masked.size,
                'mean': masked.mean(),
                'std': masked.std(),
                'unq': numpy.unique(masked.flatten()).size,
                'range': masked.max() - masked.min(),
                'cv': masked.var()}
        else:
            feature_stats = {
                'min': 0,
                'max': 0,
                'sum': 0,
                'count': 0,
                'mean': 0,
                'std': 0,
                'unq': 0,
                'range': 0,
                'cv': 0}
    else:
        banddataraster_1 = raster.GetRasterBand(1)
        banddataraster_2 = raster.GetRasterBand(2)
        banddataraster_3 = raster.GetRasterBand(3)
        dataraster_1 = banddataraster_1.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint16)
        dataraster_2 = banddataraster_2.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint16)
        dataraster_3 = banddataraster_3.ReadAsArray(xoff, yoff, xcount, ycount).astype(numpy.uint16)

        masked_1 = numpy.extract(numpy.logical_not(datamask),dataraster_1)
        masked_2 = numpy.extract(numpy.logical_not(datamask),dataraster_2)
        masked_3 = numpy.extract(numpy.logical_not(datamask),dataraster_3)

        if masked_1.size !=0 and masked_2.size !=0 and masked_3.size !=0:
        # Calculate statistics of zonal raster
            feature_stats = {
                'min_b1': masked_1.min(),
                'min_b2': masked_2.min(),
                'min_b3': masked_3.min(),
                'max_b1': masked_1.max(),
                'max_b2': masked_2.max(),
                'max_b3': masked_3.max(),
                'sum_b1': masked_1.sum(),
                'sum_b2': masked_2.sum(),
                'sum_b3': masked_3.sum(),
                'count_b1': masked_1.size,
                'count_b2': masked_2.size,
                'count_b3': masked_3.size,
                'mean_b1': masked_1.mean(),
                'mean_b2': masked_2.mean(),
                'mean_b3': masked_3.mean(),
                'std_b1': masked_1.std(),
                'std_b2': masked_2.std(),
                'std_b3': masked_3.std(),
                'unq_b1': numpy.unique(masked_1.flatten()).size,
                'unq_b2': numpy.unique(masked_2.flatten()).size,
                'unq_b3': numpy.unique(masked_3.flatten()).size,
                'range_b1': masked_1.max() - masked_1.min(),
                'range_b2': masked_2.max() - masked_2.min(),
                'range_b3': masked_3.max() - masked_3.min(),
                'cv_b1': masked_1.var(),
                'cv_b2': masked_2.var(),
                'cv_b3': masked_3.var()}
        else:
            feature_stats = {
                'min_b1': 0,
                'min_b2': 0,
                'min_b3': 0,
                'max_b1': 0,
                'max_b2': 0,
                'max_b3': 0,
                'sum_b1': 0,
                'sum_b2': 0,
                'sum_b3': 0,
                'count_b1': 0,
                'count_b2': 0,
                'count_b3': 0,
                'mean_b1': 0,
                'mean_b2': 0,
                'mean_b3': 0,
                'std_b1': 0,
                'std_b2': 0,
                'std_b3': 0,
                'unq_b1': 0,
                'unq_b2': 0,
                'unq_b3': 0,
                'range_b1': 0,
                'range_b2': 0,
                'range_b3': 0,
                'cv_b1': 0,
                'cv_b2': 0,
                'cv_b3': 0}


    return feature_stats

def loop_zonal_stats(input_zone_polygon, input_value_raster,category):
    shp = ogr.Open(input_zone_polygon, update=1)
    rows_temp,cols_temp,nbands_temp,geot_temp,proj_temp = read_image_parameters(input_value_raster)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    if nbands_temp == 1:
        lyr.CreateField(ogr.FieldDefn("min_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("max_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("sum_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("count_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("mean_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("std_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("unq_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("range_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("cv_"+category, ogr.OFTReal))
    else:
        lyr.CreateField(ogr.FieldDefn("min_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("min_b2_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("min_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("max_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("max_b2"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("max_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("sum_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("sum_b2_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("sum_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("count_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("count_b2_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("count_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("mean_b1_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("mean_b2_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("mean_b3_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("std_b1_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("std_b2_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("std_b3_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("unq_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("unq_b2_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("unq_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("range_b1_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("range_b2_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("range_b3_"+category, ogr.OFTInteger))
        lyr.CreateField(ogr.FieldDefn("cv_b1_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("cv_b2_"+category, ogr.OFTReal))
        lyr.CreateField(ogr.FieldDefn("cv_b3_"+category, ogr.OFTReal))
    
    for feature_id in featList:
        print str(feature_id) + ' of ' + str(lyr.GetFeatureCount())
        feat = lyr.GetFeature(feature_id)
        stats = zonal_stats(feat, input_zone_polygon, input_value_raster)
        if nbands_temp == 1:
            a = (stats["min"], stats["max"], stats["sum"], stats["count"], stats["mean"], stats["std"], stats["unq"], stats["range"], stats["cv"])

            feat.SetField("min_"+category,int(a[0]))
            feat.SetField("max_"+category,int(a[1]))
            feat.SetField("sum_"+category,int(a[2]))
            feat.SetField("count_"+category,int(a[3]))
            feat.SetField("mean_"+category,float(a[4]))
            feat.SetField("std_"+category,float(a[5]))
            feat.SetField("unq_"+category,int(a[6]))
            feat.SetField("range_"+category,int(a[7]))
            feat.SetField("cv_"+category,float(a[8]))
        else:
            a = (stats["min_b1"], stats["min_b2"], stats["min_b3"], stats["max_b1"], stats["max_b2"], stats["max_b3"], stats["sum_b1"], stats["sum_b2"], stats["sum_b3"], stats["count_b1"], stats["count_b2"], stats["count_b3"], stats["mean_b1"], stats["mean_b2"], stats["mean_b3"], stats["std_b1"], stats["std_b2"], stats["std_b3"], stats["unq_b1"], stats["unq_b2"], stats["unq_b3"], stats["range_b1"], stats["range_b2"], stats["range_b3"], stats["cv_b1"], stats["cv_b2"], stats["cv_b3"])
            
            feat.SetField("min_b1_"+category,int(a[0]))
            feat.SetField("min_b2_"+category,int(a[1]))
            feat.SetField("min_b3_"+category,int(a[2]))
            feat.SetField("max_b1_"+category,int(a[3]))
            feat.SetField("max_b2_"+category,int(a[4]))
            feat.SetField("max_b3_"+category,int(a[5]))
            feat.SetField("sum_b1_"+category,int(a[6]))
            feat.SetField("sum_b2_"+category,int(a[7]))
            feat.SetField("sum_b3_"+category,int(a[8]))
            feat.SetField("count_b1_"+category,int(a[9]))
            feat.SetField("count_b2_"+category,int(a[10]))
            feat.SetField("count_b3_"+category,int(a[11]))
            feat.SetField("mean_b1_"+category,float(a[12]))
            feat.SetField("mean_b2_"+category,float(a[13]))
            feat.SetField("mean_b3_"+category,float(a[14]))
            feat.SetField("std_b1_"+category,float(a[15]))
            feat.SetField("std_b2_"+category,float(a[16]))
            feat.SetField("std_b3_"+category,float(a[17]))
            feat.SetField("unq_b1_"+category,int(a[18]))
            feat.SetField("unq_b2_"+category,int(a[19]))
            feat.SetField("unq_b3_"+category,int(a[20]))
            feat.SetField("range_b1_"+category,int(a[21]))
            feat.SetField("range_b2_"+category,int(a[22]))
            feat.SetField("range_b3_"+category,int(a[23]))
            feat.SetField("cv_b1_"+category,float(a[24]))
            feat.SetField("cv_b2_"+category,float(a[25]))
            feat.SetField("cv_b3_"+category,float(a[26]))

        '''
        feat.SetField("min_"+category,stats["min"])
        feat.SetField("max_"+category,stats["max"])
        feat.SetField("sum_"+category,stats["sum"])
        feat.SetField("count_"+category,stats["count"])
        feat.SetField("mean_"+category,stats["mean"])
        feat.SetField("std_"+category,stats["std"])
        feat.SetField("unique_"+category,stats["unique"])
        feat.SetField("range_"+category,stats["range"])
        feat.SetField("cv_"+category,stats["cv"])
        '''
        

        lyr.SetFeature(feat)


class Donut(object):

    def __init__(self,x_min,x_max,y_min,y_max,resize):
        self.x_min,self.x_max,self.y_min,self.y_max,self.resize = x_min,x_max,y_min,y_max,resize

    def add(self):
        outWindow = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
        outWindow.AddPoint(self.x_min-self.resize, self.y_max+self.resize)
        outWindow.AddPoint(self.x_max+self.resize, self.y_max+self.resize)
        outWindow.AddPoint(self.x_max+self.resize, self.y_min-self.resize)
        outWindow.AddPoint(self.x_min-self.resize, self.y_min-self.resize)
        outWindow.CloseRings()
        innerWindow = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
        innerWindow.AddPoint(self.x_min+self.resize, self.y_max-self.resize)
        innerWindow.AddPoint(self.x_max-self.resize, self.y_max-self.resize)
        innerWindow.AddPoint(self.x_max-self.resize, self.y_min+self.resize)
        innerWindow.AddPoint(self.x_min+self.resize, self.y_min+self.resize)
        innerWindow.CloseRings()
        poly = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
        poly.AddGeometry(outWindow)
        poly.AddGeometry(innerWindow)
        return poly

def clip_mask(input_raster,input_vector,output_raster):
    clip_rectangular(input_raster,0,input_vector,os.path.splitext(input_raster)[0]+'_mask.tif',mask=True,resize=RSZ)
    clip_rectangular(input_raster,0,input_vector,os.path.splitext(input_raster)[0]+'_mask_clipped.tif',resize=RSZ)
    mask = read_image(os.path.splitext(input_raster)[0]+'_mask.tif',0,0)[0]
    image = read_image(os.path.splitext(input_raster)[0]+'_mask_clipped.tif',0,0)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(os.path.splitext(input_raster)[0]+'_mask_clipped.tif')
    band_list = list()
    for b in range (0,nbands):
        masked = numpy.ma.masked_array(mask, image[b])
        band_list.append(masked)
    write_image(band_list,0,0,output_raster,rows,cols,geo_transform,projection)

def buffer_2meters(inputShape,panchromatic,panchromatic_output):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    inputDS = driver.Open(inputShape)
    inputLayer = inputDS.GetLayer()
    inputFeaturesCount = inputLayer.GetFeatureCount()
    output_shape_name =  os.path.splitext(inputShape)[0]+'_buffered.shp'
    if os.path.isfile(output_shape_name): os.remove(output_shape_name)
    outDS = driver.CreateDataSource(output_shape_name)
    outLayer = outDS.CreateLayer("Donuts", inputLayer.GetSpatialRef(), ogr.wkbPolygon)
    outLayer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    outLayer.CreateField(ogr.FieldDefn("Mask", ogr.OFTInteger))
    for i in range(inputFeaturesCount):
        print "{} di {} features".format(i+1,inputFeaturesCount)
        inputFeature = inputLayer.GetFeature(i)
        maker = WindowsMaker(inputFeature,2)
        donutFeature = maker.make_feature(polygon=Donut)
        outLayer.CreateFeature(donutFeature)
        outFeature = outLayer.GetFeature(i)
        outFeature.SetField('Mask',1)
        outLayer.SetFeature(outFeature)
    del outDS
    clip_mask(panchromatic,output_shape_name,panchromatic_output)

import cv2 
from matplotlib import pyplot as plt
def canny(input_raster,output_raster):
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    img = read_image(input_raster,np.uint8,0)
    edges = cv2.Canny(img[0],1,1)
    write_image([edges],np.uint8,0,output_raster,rows,cols,geo_transform,projection)

def slope(input_dem,output_slope_map,input_type='pansharp'):
    if input_type == 'panchromatic':
        executeCmd("gdaldem slope {} {}".format(input_dem, output_slope_map))
    else:
        band_list = read_image(input_dem,np.uint16,0)
        rows_ps,cols_ps,nbands_ps,geot_ps,proj_ps = read_image_parameters(input_dem)

        write_image([band_list[0]],0,0,input_dem[:-4]+'_b1.tif',rows_ps,cols_ps,geot_ps,proj_ps)
        write_image([band_list[1]],0,0,input_dem[:-4]+'_b2.tif',rows_ps,cols_ps,geot_ps,proj_ps)
        write_image([band_list[2]],0,0,input_dem[:-4]+'_b3.tif',rows_ps,cols_ps,geot_ps,proj_ps)

        executeCmd("gdaldem slope {} {}".format(input_dem[:-4]+'_b1.tif', output_slope_map[:-4]+'_b1.tif'))
        executeCmd("gdaldem slope {} {}".format(input_dem[:-4]+'_b2.tif', output_slope_map[:-4]+'_b2.tif'))
        executeCmd("gdaldem slope {} {}".format(input_dem[:-4]+'_b3.tif', output_slope_map[:-4]+'_b3.tif'))

if __name__ == "__main__":
    main()
