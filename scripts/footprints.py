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
import scipy as sp
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
    pansharp_file = str(arg.pansharp_file)
    training_set = str(arg.training_set)
    training_attribute = str(arg.training_attribute)
    building_classes = map(int, arg.classes)
    output_shape = str(arg.output_shape)
    enable_smooth_filter = arg.optimazer
    print pansharp_file,training_set,training_attribute,building_classes,output_shape,enable_smooth_filter
    footprints(pansharp_file,training_set,training_attribute,building_classes,output_shape,enable_smooth_filter=enable_smooth_filter)

def args():
    parser = argparse.ArgumentParser(description='Calculate Height')
    parser.add_argument("pansharp_file", help="????")
    parser.add_argument("training_set", help="????")
    parser.add_argument("training_attribute", help="????")
    parser.add_argument("output_shape", help="????")
    parser.add_argument("-c", "--classes", nargs="+", help="????")
    parser.add_argument("--optimizer", default=False, const=True, nargs='?', help="????")
    args = parser.parse_args()
    return args

def footprints(pansharp_file,training_set,training_attribute,building_classes,output_shape,enable_smooth_filter=True):
    status = Bar(500, "1/2 Converting shapefile in raster.")
    #Apply smooth filter to original image
    if enable_smooth_filter == True:
        status(1,status="(1/5) Smooth filter...this may take a while")
        print 'Smooth filter...this may take a while'
        smooth_filter_otb(pansharp_file,pansharp_file[:-4]+'_smooth.tif',30)
        process_file = pansharp_file[:-4]+'_smooth.tif'
    else:
        process_file = pansharp_file
    status(100,status="(2/5) Supervised classification...")
    print 'Supervised classification...'
    train_classifier_otb([process_file],[training_set],pansharp_file[:-4]+'_svm.txt','svm',training_attribute)
    supervised_classification_otb(process_file,pansharp_file[:-4]+'_svm.txt',pansharp_file[:-4]+'_svm.tif')
    status(200,status="(3/5) Conversion to shapefile...")
    print 'Conversion to shapefile...'
    if os.path.isfile(pansharp_file[:-4]+'_svm.shp'): os.remove(pansharp_file[:-4]+'_svm.shp')
    rast2shp(pansharp_file[:-4]+'_svm.tif',pansharp_file[:-4]+'_svm.shp')
    status(300,status="(4/5) Area and Class filtering...")
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
        if dn in building_classes and area > 25 and area < 6000 :
            outfeature = osgeo.ogr.Feature(feature_def)
            outfeature.SetGeometry(geom)
            outfeature.SetField('DN',dn)
            outfeature.SetField('Area',area)
            outlayer.CreateFeature(outfeature)
            outfeature.Destroy()
        infeature = inlayer.GetNextFeature()
    infile.Destroy()
    status(400,status="(5/5) Conversion to shapefile...")
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
    status(500,status="Finished...")
    outfile.Destroy()
    final_file.Destroy()
    

if __name__ == "__main__":
    main()
    