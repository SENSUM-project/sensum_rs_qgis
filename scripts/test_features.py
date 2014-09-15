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
from sensum_library.multi import *

def main():
    warnings.filterwarnings("ignore")
    arg = args()
    input_file = str(arg.input_file)
    segmentation_shape = str(arg.segmentation_shape)
    output_shape = str(arg.output_shape)
    field = str(arg.field)
    spectrals = ["max_br", "mean", "min_br", "mode", "ndvi_std", "ndvi_mean", "std", "weigh_br"]
    indexes_list_spectral = [spectral for spectral in spectrals if getattr(arg, spectral)]
    textures = ["ASM", "contrast", "correlation", "dissimilarity", "energy", "homogeneity"]
    indexes_list_texture = [texture for texture in textures if getattr(arg, texture)]
    if os.path.isfile(output_shape): os.remove(output_shape)
    if arg.multi:
        test_features_multi(input_file, segmentation_shape, output_shape, indexes_list_spectral, indexes_list_texture,field)
    else:
        test_features(input_file, segmentation_shape, output_shape, indexes_list_spectral, indexes_list_texture,field)

def args():
    parser = argparse.ArgumentParser(description='Calculate Features')
    parser.add_argument("input_file", help="????")
    parser.add_argument("segmentation_shape", help="????")
    parser.add_argument("output_shape", help="????")
    parser.add_argument('field', help="????")
    parser.add_argument("--max_br", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--mean", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--min_br", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--mode", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--ndvi_std", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--ndvi_mean", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--std", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--weigh_br", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--ASM", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--contrast", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--correlation", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--dissimilarity", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--energy", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--homogeneity", default=False, const=True, nargs='?', help="????")
    parser.add_argument("--multi", default=False, const=True, nargs='?', help="????")
    args = parser.parse_args()
    return args

def test_features(input_file,segmentation_shape,output_shape,indexes_list_spectral,indexes_list_texture,field):
    start_time = time.time()
    ndvi_comp = []
    wb_comp = []
    #Read original image - base layer
    input_list = read_image(input_file,np.uint16,0)
    #input_list_tf = read_image(input_file,np.uint8,0) #different data type necessary for texture features
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_file)
    input_list_tf = linear_quantization(input_list,64)
    print indexes_list_spectral
    #Conversion of the provided segmentation shapefile to raster for further processing
    shp2rast(segmentation_shape, segmentation_shape[:-4]+'.TIF', rows, cols, field)
    seg_list = read_image(segmentation_shape[:-4]+'.TIF',np.int32,0)
    if (('ndvi_mean' in indexes_list_spectral) or ('ndvi_std' in indexes_list_spectral)) and nbands > 3:
        ndvi = (input_list[3].astype(float)-input_list[2].astype(float)) / (input_list[3].astype(float)+input_list[2].astype(float))
        ndvi_comp = [s for s in indexes_list_spectral if 'ndvi_mean' in s or 'ndvi_std' in s]
    if 'weigh_br' in indexes_list_spectral:
        band_sum = np.zeros((rows,cols))
        for b in range(0,nbands):
            band_sum = band_sum + input_list[b]
        wb_comp = [s for s in indexes_list_spectral if 'weigh_br' in s]
    ind_list_spectral = [s for s in indexes_list_spectral if 'ndvi_mean' not in s and 'ndvi_std' not in s and 'weigh_br' not in s]

    #print ind_list_spectral
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
            
        for sp in range(0,len(indexes_list_texture)):
            if len(indexes_list_texture[sp]+str(b)) > 10:
                cut = len(indexes_list_texture[sp]+str(b)) - 10 
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp][:-cut] + str(b), osgeo.ogr.OFTReal)
            else:
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp] + str(b), osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
        
    if ndvi_comp:
        for nd in range(0,len(ndvi_comp)):
            field_def = osgeo.ogr.FieldDefn(ndvi_comp[nd], osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
    if wb_comp:
        field_def = osgeo.ogr.FieldDefn(wb_comp[0], osgeo.ogr.OFTReal)
        outlayer.CreateField(field_def)
          
    feature_def = outlayer.GetLayerDefn()
    n_feature = inlayer.GetFeatureCount()
    i = 1
    #loop through segments
    status = Bar(n_feature, "Computing Features...")
    while infeature:
        status(i+1)
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
        if len(ind_list_spectral) > 0 or len(indexes_list_texture)>0:
            for b in range(1,nbands+1):
                if len(ind_list_spectral) > 0:
                    spectral_list = spectral_segments(input_list[b-1], dn, seg_list[0], ind_list_spectral, nbands)
                    for si in range(0,len(ind_list_spectral)):
                        outfeature.SetField(ind_list_spectral[si] + str(b),float(spectral_list[si]))
            
                if len(indexes_list_texture)>0:
                    texture_list = texture_segments(input_list_tf[b-1],dn,seg_list[0],indexes_list_texture)
                    for sp in range(0,len(indexes_list_texture)):
                        if len(indexes_list_texture[sp]+str(b)) > 10:
                            cut = len(indexes_list_texture[sp]+str(b)) - 10
                            outfeature.SetField(indexes_list_texture[sp][:-cut] + str(b),float(texture_list[sp]))
                        else:
                            outfeature.SetField(indexes_list_texture[sp] + str(b),float(texture_list[sp]))
        if ndvi_comp:
            ndvi_list = spectral_segments(ndvi, dn, seg_list[0], ndvi_comp, nbands)
            if len(ndvi_list) > 0:
                for nd in range(0,len(ndvi_comp)):
                    outfeature.SetField(ndvi_comp[nd],ndvi_list[nd])
        if wb_comp:
            wb = spectral_segments(band_sum, dn, seg_list[0], wb_comp, nbands)
            if len(wb) > 0:
                outfeature.SetField(wb_comp[0],wb[0])

        outlayer.CreateFeature(outfeature)
        outfeature.Destroy()
        infeature = inlayer.GetNextFeature()
    shutil.copyfile(segmentation_shape[:-4]+'.prj', output_shape[:-4]+'.prj')
    print 'Output created: ' + output_shape
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()
    end_time = time.time()
    print 'Total time = ' + str(end_time-start_time)

class Task(object):

    def __init__(self, ind_list_spectral, indexes_list_texture,indexes_list_spectral, cut, ndvi_comp, wb_comp,nbands, input_list, input_list_tf, seg_list, n_feature, dn,index):

        self.ind_list_spectral = ind_list_spectral
        self.indexes_list_texture = indexes_list_texture
        self.indexes_list_spectral = indexes_list_spectral
        self.cut = cut
        self.ndvi_comp = ndvi_comp
        self.wb_comp = wb_comp
        self.index = index

        self.nbands = nbands
        self.input_list = input_list
        self.input_list_tf = input_list_tf
        self.seg_list = seg_list
        self.n_feature = n_feature
        self.dn = dn

    def __call__(self):
        result = []
        if len(self.ind_list_specral) > 0 or len(self.indexes_list_texture)>0:
            for b in range(1,self.nbands+1):
                if len(self.ind_list_specral) > 0:
                    spectral_list = spectral_segments(self.input_list[b-1], self.dn, self.seg_list[0], self.ind_list_spectral, self.nbands)
                    for si in range(0,len(self.indexes_list_spectral)):
                        result.append([self.indexes_list_spectral[si] + str(b),spectral_list[si]])
                if len(self.indexes_list_texture)>0:
                    texture_list = texture_segments(self.input_list_tf[b-1],self.dn,self.seg_list[0],self.indexes_list_texture)
                    for sp in range(0,len(self.indexes_list_texture)):
                        if len(self.indexes_list_texture[sp]+str(b)) > 10:
                            result.append([self.indexes_list_texture[sp][:-self.cut] + str(b),texture_list[sp]])
                        else:
                            result.append([self.indexes_list_texture[sp] + str(b),texture_list[sp]])
            if self.ndvi_comp:
                ndvi_list = spectral_segments(self.input_list[b-1], self.dn, self.seg_list[0], self.ndvi_comp, self.nbands)
                for nd in range(0,len(self.ndvi_comp)):
                    result.append([self.ndvi_comp[nd],ndvi_list[nd]])
            if self.wb_comp:
                wb = spectral_segments(self.input_list[b-1], self.dn, self.seg_list[0], self.wb_comp, self.nbands)
                result.append([self.wb_comp[0],wb[0]])
        #print str(self.index+1) + ' of ' + str(self.n_feature)
#        print result
        return result,self.index

    def __str__(self):
        return str('')

def test_features_multi(input_file,segmentation_shape,output_shape,indexes_list_spectral,indexes_list_texture,field="DN"):
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
        ndvi = (input_list[3].astype(float)-input_list[2].astype(float)) / (input_list[3].astype(float)+input_list[2].astype(float))
        ndvi_comp = [s for s in indexes_list_spectral if 'ndvi_mean' in s or 'ndvi_std' in s]
    if 'weigh_br' in indexes_list_spectral:
        band_sum = np.zeros((rows,cols))
        for b in range(0,nbands):
            band_sum = band_sum + input_list[b]
        wb_comp = [s for s in indexes_list_spectral if 'weigh_br' in s]
    #to fix  
    ind_list_spectral = [s for s in indexes_list_spectral if 'ndvi_mean' not in s and 'ndvi_std' not in s and 'weigh_br' not in s]
    #print ind_list_spectral
    #read input shapefile
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile=driver_shape.Open(segmentation_shape,0)
    inlayer=infile.GetLayer()
    #create output shapefile 
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
            
        for sp in range(0,len(indexes_list_texture)):
            if len(indexes_list_texture[sp]+str(b)) > 10:
                cut = len(indexes_list_texture[sp]+str(b)) - 10 
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp][:-cut] + str(b), osgeo.ogr.OFTReal)
            else:
                field_def = osgeo.ogr.FieldDefn(indexes_list_texture[sp] + str(b), osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
        
    if ndvi_comp:
        for nd in range(0,len(ndvi_comp)):
            field_def = osgeo.ogr.FieldDefn(ndvi_comp[nd], osgeo.ogr.OFTReal)
            outlayer.CreateField(field_def)
    if wb_comp:
        field_def = osgeo.ogr.FieldDefn(wb_comp[0], osgeo.ogr.OFTReal)
        outlayer.CreateField(field_def)

    feature_def = outlayer.GetLayerDefn()
    n_feature = inlayer.GetFeatureCount()
    n_division = 1000
#    n_feature = 13
    j = 0
    status = Bar(n_feature, "Testing Features.")
    while 1:
        multiprocess = Multi()
        while j<n_feature:
            status((j+1)/2)
            infeature = inlayer.GetFeature(j)
            dn = infeature.GetField(field)
            multiprocess.put(Task(ind_list_spectral, indexes_list_texture,indexes_list_spectral, cut, ndvi_comp, wb_comp ,nbands, input_list, input_list_tf, seg_list, n_feature, dn, j))
            j = j + 1
            if j%n_division == 0 :
                break
            elif j == n_feature :
                n_division = j%n_division
                break
        multiprocess.kill()
        matrix = []
        for i in range(j-n_division,j):
            status((j+1)/2+(i+1)/2)
            result, index = multiprocess.result()
            matrix.append([index,result])
        del multiprocess
        #sorting results
        #matrix = sorted(matrix, key=lambda sort: sort[0])
        for i in range(len(matrix)):
            infeature = inlayer.GetFeature(j-n_division+i)
            # get the input geometry
            geom = infeature.GetGeometryRef()
            dn = infeature.GetField(field)
            # create a new feature
            outfeature = osgeo.ogr.Feature(feature_def)
            # set the geometry and attribute
            outfeature.SetGeometry(geom)
            outfeature.SetField(field,dn)
            index = matrix[i][0] 
            result = matrix[i][1]
            for n in range(len(result)):
                outfeature.SetField(result[n][0],float(result[n][1]))
            outlayer.CreateFeature(outfeature)
            outfeature.Destroy()
        del matrix
        if j == n_feature:
            break
    shutil.copyfile(segmentation_shape[:-4]+'.prj', output_shape[:-4]+'.prj')
    print 'Output created: ' + output_shape
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()
    end_time = time.time()
    print 'Total time = ' + str(end_time-start_time)    

if __name__ == "__main__":
    main()
    