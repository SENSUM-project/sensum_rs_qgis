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

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    input_buildings = str(arg.input_buildings)
    output_shape = str(arg.output_shape)
    if os.path.isfile(output_shape): os.remove(output_shape)
    regularity(input_buildings, 0.5, 0.5, output_shape)

def args():
    parser = argparse.ArgumentParser(description='Calculate Height')
    parser.add_argument("input_buildings", help="????")
    parser.add_argument("output_shape", help="????")
    args = parser.parse_args()
    return args

def regularity(buildingShape, pixelWidth, pixelHeight, outShape):
    
    '''Function for calculate regularity
    
    :param buildingShape: path of buildings shapefile (str)
    :param pixelWidth: value in meters of length of 1 pixel (float)
    :param pixelHeight: value in meters of length of 1 pixel (float)
    :param outputShape: path of output shapefile (char)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
    buildingDS = driver.Open(buildingShape)
    buildingLayer = buildingDS.GetLayer()
    buildingFeaturesCount = buildingLayer.GetFeatureCount()
    outDS = driver.CreateDataSource(outShape)
    outDS.CopyLayer(buildingLayer,"buildings")
    outLayer = outDS.GetLayer()
    regularity_def = osgeo.ogr.FieldDefn('Regularity', osgeo.ogr.OFTString)
    alignment_def = osgeo.ogr.FieldDefn('Alignment', osgeo.ogr.OFTInteger)
    #perimeter_def = osgeo.ogr.FieldDefn('Perimeter', osgeo.ogr.OFTReal)
    #length_a_def = osgeo.ogr.FieldDefn('LengthA', osgeo.ogr.OFTReal)
    #length_b_def = osgeo.ogr.FieldDefn('LengthB', osgeo.ogr.OFTReal)
    #regnew_def = osgeo.ogr.FieldDefn('Reg_new', osgeo.ogr.OFTString)
    outLayer.CreateField(regularity_def)
    outLayer.CreateField(alignment_def)
    
    status = Bar(buildingFeaturesCount, "Regularity")
    for i in range(buildingFeaturesCount):
        status(i+1)
        #Convert Shape to Raster
        try:
            buildingFeature = buildingLayer.GetFeature(i)
        except:
            i = i+1
            buildingFeature = buildingLayer.GetFeature(i)
        x_min, x_max, y_min, y_max = WindowsMaker(buildingFeature).make_coordinates()
        cols = int(math.ceil(float((x_max-x_min)) / float(pixelWidth))) #definition of the dimensions according to the extent and pixel resolution
        rows = int(math.ceil(float((y_max-y_min)) / float(pixelHeight)))
        rasterDS = gdal.GetDriverByName('MEM').Create("", cols,rows, 1, GDT_UInt16)
        rasterDS.SetGeoTransform((x_min, pixelWidth, 0,y_max, 0, -pixelHeight))
        rasterDS.SetProjection(buildingLayer.GetSpatialRef().ExportToWkt())
        gdal.RasterizeLayer(rasterDS,[1], buildingLayer,burn_values=[1])
        #Make Band List
        inband = rasterDS.GetRasterBand(1) 
        band_list = inband.ReadAsArray().astype(np.uint16)
        #Process Raster with Scipy
        '''
        w = ndimage.morphology.binary_fill_holes(band_list)
        band_list = ndimage.binary_opening(w, structure=np.ones((3,3))).astype(np.uint16)
        '''
        #Calculate regularity
        alpha,length_a,length_b = building_alignment(band_list)
        regularity = str(building_regularity(length_a,length_b))
        #Fill field
        outFeature = outLayer.GetFeature(i)
        #geom = outFeature.GetGeometryRef()
        #perimeter = geom.Boundary().Length()
        #area = geom.Area()
        #delta = math.sqrt((float(perimeter)*float(perimeter)/float(4)-4*area))
        #length_a_new = float((float(perimeter)/float(2)) + delta) / float(2)
        #length_b_new = float((float(perimeter)/float(2)) - delta) / float(2)
        '''
        if length_a_new > length_b_new:
            if float(length_a_new)/float(length_b_new) > 4: reg_label = 'irregular'
            else: reg_label = 'regular'
        else:
            if float(length_b_new)/float(length_a_new) > 4: reg_label = 'irregular'
            else: reg_label = 'regular'
        '''
        outFeature.SetField('Regularity',regularity)
        outFeature.SetField('Alignment',int(alpha))
        #outFeature.SetField('Perimeter',float(perimeter))
        #outFeature.SetField('LengthA',float(length_a_new))
        #outFeature.SetField('LengthB',float(length_b_new))
        #outFeature.SetField('Reg_new',str(reg_label))
        outLayer.SetFeature(outFeature)
    buildingDS.Destroy()
    return outDS


if __name__ == "__main__":
    main()
