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
    buildingShape = str(arg.buildingShape)
    radius = float(arg.radius)
    outputShape = str(arg.outputShape)
    if os.path.isfile(outputShape): os.remove(outputShape)
    density(buildingShape,radius,outputShape=outputShape)

def args():
    parser = argparse.ArgumentParser(description='Calculate Density')
    parser.add_argument("buildingShape", help="????")
    parser.add_argument("radius", help="????")
    parser.add_argument("outputShape", help="????")
    args = parser.parse_args()
    return args

def density(buildingShape,radius,outputShape):
    '''Function for assign density value into each building.                                                             \
    Build new dataset (or shapefile if outputShape argument declared) with field N_Building and Density which contains   \
    respectively the number of building into the expressed radius and density parameter expressed in number of building  \
    for meter.

    :param buildingShape: path of buildings shapefile (str)
    :param radius: radius of circle maked around each building, expressed in coordinates as unit of measure (float)
    :param outputShape: path of output shapefile (char)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    #get layers
    driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
    buildingsDS = driver.Open(buildingShape)
    buildingsLayer = buildingsDS.GetLayer()
    buindingsFeaturesCount = buildingsLayer.GetFeatureCount()
    if outputShape == '':
        driver = osgeo.ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outputShape)
    #copy the buildings layer and use it as output layer
    outDS.CopyLayer(buildingsLayer,"")
    outLayer = outDS.GetLayer()
    #add fields 'Density' to outLayer
    fldDef = osgeo.ogr.FieldDefn('N_Building', osgeo.ogr.OFTInteger)
    outLayer.CreateField(fldDef)
    fldDef = osgeo.ogr.FieldDefn('Density', osgeo.ogr.OFTReal)
    outLayer.CreateField(fldDef)
    #loop into building features goint to make a window around each features and taking how many buildings are there around
    status = Bar(buindingsFeaturesCount)
    for i in range(buindingsFeaturesCount):
        status(i+1)
        buildingFeature = outLayer.GetFeature(i)
        #make a spatial layer and get the layer
        maker = WindowsMaker(buildingFeature)
        centroid = buildingFeature.GetGeometryRef().Centroid()
        maker.make_feature(geom=CircleDensity(centroid,radius).add())
        area = radius**2 * math.pi
        spatialDS = maker.get_shapeDS(buildingsLayer)
        spatialLayer = spatialDS.GetLayer()
        spatialFeature = spatialLayer.GetNextFeature()
        sum_area = 0.0
        while spatialFeature:
            #area_ft = spatialFeature.GetField("Area")
            area_ft = spatialFeature.GetGeometryRef().Area()
            sum_area = area_ft + sum_area
            spatialFeature = spatialLayer.GetNextFeature()
        spatialLayerFeatureCount = spatialLayer.GetFeatureCount() -1 #(-1) for remove itself
        outFeature = outLayer.GetFeature(i)
        outFeature.SetField("N_Building",spatialLayerFeatureCount)
        if spatialLayerFeatureCount:
            outFeature.SetField("Density",float(sum_area/area))
        else:
            outFeature.SetField("Density",0)
        outLayer.SetFeature(outFeature)
    return outDS

if __name__ == "__main__":
    main()
