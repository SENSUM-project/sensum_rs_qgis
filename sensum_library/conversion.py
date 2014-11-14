'''
.. module:: conversion
   :platform: Unix, Windows
   :synopsis: This module includes functions related to conversions between different data types and reference systems.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia
'''
'''
---------------------------------------------------------------------------------
                                conversion.py
---------------------------------------------------------------------------------
Created on May 13, 2013
Last modified on Mar 18, 2014
---------------------------------------------------------------------------------
Project: Framework to integrate Space-based and in-situ sENSing for dynamic 
         vUlnerability and recovery Monitoring (SENSUM)

Co-funded by the European Commission under FP7 (Seventh Framework Programme)
THEME [SPA.2012.1.1-04] Support to emergency response management
Grant agreement no: 312972

---------------------------------------------------------------------------------
License: This file is part of SensumTools.

    SensumTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SensumTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SensumTools.  If not, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------------------
'''

import config
import os
import sys
import osgeo.osr
import osgeo.ogr
import osgeo.gdal
import math
from gdalconst import *
import numpy as np
import otbApplication
if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'


def data_type2gdal_data_type(data_type):
    
    '''Conversion from numpy data type to GDAL data type
    
    :param data_type: numpy type (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
    :returns: corresponding GDAL data type
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 
    #Function needed when it is necessary to write an output file
    if data_type == np.uint16:
        return GDT_UInt16
    if data_type == np.uint8:
        return GDT_Byte
    if data_type == np.int32:
        return GDT_Int32
    if data_type == np.float32:
        return GDT_Float32
    if data_type == np.float64:
        return GDT_Float64
    

def read_image(input_raster,data_type,band_selection):
    
    '''Read raster using GDAL
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type).
    :param band_selection: number associated with the band to extract (0: all bands, 1: blue, 2: green, 3:red, 4:infrared) (integer).
    :returns:  a list containing the desired bands as ndarrays (list of arrays).
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 
    
    #TODO: Why not restrict this function to return band_list only? Would make it more clear and not redundant with Read_Image_Parameters.
    #TODO: You use as default import type uint16 but for export of images you use gdt_float32. 
    #TODO: Is this the general function to make rasters available to functions? How do you deal with GDAL to OpenCV matrices?
    band_list = []
    
    if data_type == 0: #most of the images (MR and HR) can be read as uint16
        data_type = np.uint16
        
    inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    cols=inputimg.RasterXSize
    rows=inputimg.RasterYSize
    nbands=inputimg.RasterCount
    
    if band_selection == 0:
        #read all the bands
        for i in range(1,nbands+1):
            inband = inputimg.GetRasterBand(i) 
            #mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
            mat_data = inband.ReadAsArray().astype(data_type)
            band_list.append(mat_data) 
    else:
        #read the single band
        inband = inputimg.GetRasterBand(band_selection) 
        mat_data = inband.ReadAsArray(0,0,cols,rows).astype(data_type)
        band_list.append(mat_data)
    
    inputimg = None    
    return band_list


def read_image_parameters(input_raster):
    
    '''Read raster parameters using GDAL
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string).
    :returns:  a list containing rows, columns, number of bands, geo-transformation matrix and projection.
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 
   
    inputimg = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    cols=inputimg.RasterXSize
    rows=inputimg.RasterYSize
    nbands=inputimg.RasterCount
    geo_transform = inputimg.GetGeoTransform()
    projection = inputimg.GetProjection()
    
    inputimg = None
    return rows,cols,nbands,geo_transform,projection


def write_image(band_list,data_type,band_selection,output_raster,rows,cols,geo_transform,projection):
   
    '''Write array to file as raster using GDAL
    
    :param band_list: list of arrays containing the different bands to write (list of arrays).
    :param data_type: numpy data type of the output image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :param band_selection: number associated with the band to write (0: all, 1: blue, 2: green, 3: red, 4: infrared) (integer)
    :param output_raster: path and name of the output raster to create (*.TIF, *.tiff) (string)
    :param rows: rows of the output raster (integer)
    :param cols: columns of the output raster (integer)
    :param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
    :param projection: projection of the output image (string)
    :returns: An output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''

    if data_type == 0:
        gdal_data_type = GDT_UInt16 #default data type
    else:
        gdal_data_type = data_type2gdal_data_type(data_type)
    
    driver = osgeo.gdal.GetDriverByName('GTiff')

    if band_selection == 0:
        nbands = len(band_list)
    else:
        nbands = 1
    outDs = driver.Create(output_raster, cols, rows,nbands, gdal_data_type)
    if outDs is None:
        print 'Could not create output file'
        sys.exit(1)
        
    if band_selection == 0:
        #write all the bands to file
        for i in range(0,nbands): 
            outBand = outDs.GetRasterBand(i+1)
            outBand.WriteArray(band_list[i], 0, 0)
    else:
        #write the specified band to file
        outBand = outDs.GetRasterBand(1)   
        outBand.WriteArray(band_list[band_selection-1], 0, 0)
    #assign geomatrix and projection
    outDs.SetGeoTransform(geo_transform)
    outDs.SetProjection(projection)
    outDs = None


def linear_quantization(input_band_list,quantization_factor):
    
    '''Quantization of all the input bands cutting the tails of the distribution
    
    :param input_band_list: list of 2darrays (list of 2darrays)
    :param quantization_factor: number of levels as output (integer)
    :returns:  list of values corresponding to the quantized bands (list of 2darray)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 12/05/2014
    '''
    band_list_q = []
    q_factor = quantization_factor - 1
    for b in range(0,len(input_band_list)):
        inmatrix = input_band_list[b].reshape(-1)
        out = np.bincount(inmatrix)
        tot = inmatrix.shape[0]
        freq = (out.astype(np.float32)/float(tot))*100 #frequency for each value
        cumfreqs = np.cumsum(freq)
    
        first = np.where(cumfreqs>1.49)[0][0] #define occurrence limits for the distribution
        last = np.where(cumfreqs>97.8)[0][0]
        input_band_list[b][np.where(input_band_list[b]>last)] = last
        input_band_list[b][np.where(input_band_list[b]<first)] = first

        k1 = float(q_factor)/float((last-first)) #k1 term of the quantization formula
        k2 = np.ones(input_band_list[b].shape)-k1*first*np.ones(input_band_list[b].shape) #k2 term of the quantization formula
        out_matrix = np.floor(input_band_list[b]*k1+k2) #take the integer part
        out_matrix2 = out_matrix-np.ones(out_matrix.shape)
        out_matrix2.astype(np.uint8)

        band_list_q.append(out_matrix2) #list of quantized 2darrays
    
    return band_list_q


def shp2rast(input_shape,output_raster,rows,cols,field_name,pixel_width=0,pixel_height=0,x_min=0,x_max=0,y_min=0,y_max=0):
    
    '''Conversion from shapefile to raster using GDAL
    
    :param input_shape: path and name of the input shapefile (*.shp) (string)
    :param output_raster: path and name of the output raster to create (*.TIF, *.tiff) (string)
    :param rows: rows of the output raster (integer)
    :param cols: columns of the output raster (integer)
    :param field_name: name of the attribute field of the shapefile used to differentiate pixels (string)
    :param pixel_width: pixel resolution x axis
    :param pixel_height: pixel resolution y axis
    :param x_min: minimum longitude (used for the geomatrix)
    :param x_max: maximum longitude
    :param y_min: minimum latitude
    :param y_max: maximum latitude (used for the geomatrix)
    :returns: An output file is created and a list with the shapefile extent is returned (x_min,x_max,y_min,y_max)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    
    Reference: http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
    '''
  
    #TODO: Explain additional arguments px_W,px_H,x_min,x_max,y_min,y_max
    
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver_shape.Open(input_shape)
    source_layer = data_source.GetLayer()
    source_srs = source_layer.GetSpatialRef()
    
    '''
    Conversion can be performed in 3 different ways according to the input shapefile:
    - providing rows and columns only -> the pixel resolution is computed using the extent of the shapefile in combination with rows and cols
        -> used when rows and columns are known and the shapefile extent is equal to the raster dimension
    - providing pixel resolution (pixel_width and pixel_height) only -> number of rows and columns is computed using the resolution and the shapefile extent
        -> used when it is necessary to force the same resolution, for example case of small subset of a raster
    - providing rows,columns,pixel resolution and extent -> no computation needed, parameters directly assigned to the output raster (used to force dimensions)
        -> used when you have to compare the shapefile with a raster but the actual extent of the shapefile is different because polygons are not touching the raster edges
    '''
    
    if x_min==0 or x_max==0 or y_min==0 or y_max==0: #case of non provided extension
        x_min, x_max, y_min, y_max = source_layer.GetExtent() #get the extent directly from the shapefile
    
    if rows!=0 and cols!=0 and pixel_width!=0 and pixel_height!=0 and x_min!=0 and y_max!=0: #case with rows, columns, pixel width and height and starting coordinates
        pixel_size_x = pixel_width
        pixel_size_y = abs(pixel_height)
        
    else:
        if rows != 0 and cols != 0: #case with rows and columns 
            pixel_size_x = float((x_max-x_min)) / float(cols) #definition of the resolution depending on the extent and dimensions in pixels
            pixel_size_y = float((y_max-y_min)) / float(rows)
        else: #case with pixel resolution
            pixel_size_x = pixel_width
            pixel_size_y = abs(pixel_height)
            cols = int(float((x_max-x_min)) / float(pixel_size_x)) #definition of the dimensions according to the extent and pixel resolution
            rows = int(float((y_max-y_min)) / float(pixel_size_y))
    if rows!=0 and cols!=0:    
        target_ds = osgeo.gdal.GetDriverByName('GTiff').Create(output_raster, cols,rows, 1, GDT_Float32)
        target_ds.SetGeoTransform((x_min, pixel_size_x, 0,y_max, 0, -pixel_size_y))
        if source_srs:
            # Make the target raster have the same projection as the source
            target_ds.SetProjection(source_srs.ExportToWkt())
        else:
            # Source has no projection (needs GDAL >= 1.7.0 to work)
            target_ds.SetProjection('LOCAL_CS["arbitrary"]')
        
        # Rasterize
        err = osgeo.gdal.RasterizeLayer(target_ds,[1], source_layer,burn_values=[0],options=["ATTRIBUTE="+field_name])
        if err != 0:
            raise Exception("error rasterizing layer: %s" % err)
        
    return x_min,x_max,y_min,y_max

def polygon2array(input_layer,pixel_width,pixel_height): 
    
    '''Conversion from polygon to array
    
    :param input_layer: layer taken from the shapefile (shapefile layer)
    :param pixel_width: pixel resolution x axis (float, positive)
    :param pixel_height: pixel resolution y axis (float, positive) 
    :returns: A matrix with the output band and the related geomatrix
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    
    Reference: http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
    '''
    
    x_min, x_max, y_min, y_max = input_layer.GetExtent()
    #print x_min, x_max, y_min, y_max
    #print pixel_width
    #print pixel_height
    x_res = int(math.ceil((x_max - x_min) / pixel_width)) #pixel x-axis resolution
    y_res = int(math.ceil((y_max - y_min) / pixel_height)) #pixel y-axis resolution
    target_ds = osgeo.gdal.GetDriverByName('MEM').Create('', x_res, y_res, GDT_Byte) #create layer in memory
    geo_transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] #geomatrix definition
    target_ds.SetGeoTransform(geo_transform)
    band = target_ds.GetRasterBand(1)
    
    # Rasterize
    osgeo.gdal.RasterizeLayer(target_ds, [1], input_layer, burn_values=[1])
    
    # Read as array
    array = band.ReadAsArray()
    target_ds = None
    input_layer = None
    return array,geo_transform


def rast2shp(input_raster,output_shape):
    
    '''Conversion from raster to shapefile using GDAL
    
    :param input_raster: path and name of the input raster (*.TIF, *.tiff) (string)
    :param output_shape: path and name of the output shapefile to create (*.shp) (string)
    :returns: An output shapefile is created 
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    
    Reference: http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    '''

    src_image = osgeo.gdal.Open(input_raster)
    src_band = src_image.GetRasterBand(1)
    projection = src_image.GetProjection()
    
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    outfile=driver_shape.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Conversion',geom_type=osgeo.ogr.wkbPolygon)
    dn = osgeo.ogr.FieldDefn('DN',osgeo.ogr.OFTInteger)
    outlayer.CreateField(dn)
    
    #Polygonize
    osgeo.gdal.Polygonize(src_band,src_band.GetMaskBand(),outlayer,0)
    
    outprj=osgeo.osr.SpatialReference(projection)
    outprj.MorphToESRI()
    file_prj = open(output_shape[:-4]+'.prj', 'w')
    file_prj.write(outprj.ExportToWkt())
    file_prj.close()
    src_image = None
    outfile = None
  

def world2pixel(geo_transform, long, lat):
    
    '''Conversion from geographic coordinates to matrix-related indexes
    
    :param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
    :param long: longitude of the desired point (float)
    :param lat: latitude of the desired point (float)
    :returns: A list with matrix-related x and y indexes (x,y)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''
    
    ulX = geo_transform[0] #starting longitude
    ulY = geo_transform[3] #starting latitude
    xDist = geo_transform[1] #x resolution
    yDist = geo_transform[5] #y resolution

    pixel_x = int((long - ulX) / xDist)
    pixel_y = int((ulY - lat) / abs(yDist))
    if pixel_x < 0 or pixel_y < 0:
        ValueError(int)
    else:
        return (pixel_x, pixel_y)


def pixel2world(geo_transform, cols, rows):
    
    '''Calculation of the geo-spatial coordinates of top-left and down-right pixel
    
    :param geo_transform: geo-transformation matrix containing coordinates and resolution of the output (array of 6 elements, float)
    :param rows: number of rows (integer)
    :param cols: number of columns (integer)
    :returns: A list with top-left and down-right coordinates
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''
    
    miny = geo_transform[3] + cols*geo_transform[4] + rows*geo_transform[5] 
    maxx = geo_transform[0] + cols*geo_transform[1] + rows*geo_transform[2]
    
    return (maxx,miny)


def utm2wgs84(easting, northing, zone):
    
    '''Conversion from UTM projection to WGS84
    
    :param easting: east coordinate (float)
    :param northing: north coordinate (float)
    :param zone: number of the utm zone (integer)
    :returns: A list with coordinates in the WGS84 system (longitude,latitude,altitude)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    
    Reference: http://monkut.webfactional.com/blog/archive/2012/5/2/understanding-raster-basic-gis-concepts-and-the-python-gdal-library/
    '''

    #TODO: Do we really need this function?
    
    utm_coordinate_system = osgeo.osr.SpatialReference()
    utm_coordinate_system.SetWellKnownGeogCS("WGS84") # Set geographic coordinate system to handle lat/lon
    is_northern = northing > 0    
    utm_coordinate_system.SetUTM(zone, is_northern)
    
    wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS() # Clone ONLY the geographic coordinate system 
    
    # create transform component
    utm_to_wgs84_geo_transform = osgeo.osr.CoordinateTransformation(utm_coordinate_system, wgs84_coordinate_system) # (, )
    lon,lat,altitude = utm_to_wgs84_geo_transform.TransformPoint(easting, northing, 0) #return lon, lat and altitude
    return lon, lat, altitude 


def reproject_shapefile(input_shape,output_shape,output_projection):
    
    '''Reproject a shapefile using the provided EPSG code
    
    :param input_shape: path and name of the input shapefile (*.shp) (string)
    :param output_shape: path and name of the output shapefile (*.shp) (string)
    :param output_projection: epsg code (integer)
    :param geometry_type: geometry type of the output ('line','polygon','point') (string)
    :returns:  an output shapefile is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 24/03/2014
    ''' 

    #TODO: It seems that you transform from a default epsg 4326 and don't allow to define the input or simply read it from the input file
    #TODO: would use only one argument to define input. 
    #driver definition for shapefile
    driver=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    
    #select input file and create an output file
    infile=driver.Open(input_shape,0)
    inlayer=infile.GetLayer()
    inprj = inlayer.GetSpatialRef() #get the original spatial reference
    if inprj == None: #if spatial reference not existing, use the default one
        inprj = 4326
    
    outprj=osgeo.osr.SpatialReference()
    outprj.ImportFromEPSG(output_projection)
    
    newcoord=osgeo.osr.CoordinateTransformation(inprj,outprj)
    
    feature=inlayer.GetNextFeature()
    gm = feature.GetGeometryRef()
    geometry_type = gm.GetGeometryName() 

    if geometry_type == 'LINE':
        type = osgeo.ogr.wkbLineString
    if geometry_type == 'POLYGON':
        type = osgeo.ogr.wkbPolygon
    if geometry_type == 'POINT':
        type = osgeo.ogr.wkbPoint 
    outfile=driver.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('rpj',geom_type=type)
    
    layer_defn = inlayer.GetLayerDefn() #get definitions of the layer
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())] #store the field names as a list of strings
    #print field_names
    for i in range(0,len(field_names)):
        field = feature.GetFieldDefnRef(field_names[i])
        outlayer.CreateField(field)
        
    # get the FeatureDefn for the output shapefile
    feature_def = outlayer.GetLayerDefn()
    inlayer.ResetReading()
    # loop through the input features
    infeature = inlayer.GetNextFeature()
    while infeature:
        # get the input geometry
        geom = infeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(newcoord)
        # create a new feature
        outfeature = osgeo.ogr.Feature(feature_def)
        # set the geometry and attribute
        outfeature.SetGeometry(geom)
        #field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        for i in range(0,len(field_names)):
            #print infeature.GetField(field_names[i])
            outfeature.SetField(field_names[i],infeature.GetField(field_names[i]))
            # add the feature to the shapefile
        outlayer.CreateFeature(outfeature)

        # destroy the features and get the next input feature
        outfeature.Destroy
        infeature.Destroy
        infeature = inlayer.GetNextFeature()

    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()

    # create the *.prj file
    outprj.MorphToESRI()
    prjfile = open(output_shape[:-4]+'.prj', 'w')
    prjfile.write(outprj.ExportToWkt())
    prjfile.close()
    
    
def split_shape(input_layer,index,option="memory",output_shape="out"):
   
    '''Extract a single feature from a shapefile
    
    :param input_layer: layer of a shapefile (shapefile layer)
    :param index: index of the feature to extract (integer)
    :param option: 'memory' or 'file' depending on the desired output (default is memory) (string)
    :param output_shape: path and name of the output shapefile (temporary file) (*.shp) (string)
    :returns:  an output shapefile is created
    :raises: AttributeError, KeyError
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    ''' 

    #TODO: Why do we need this function? Does not seems like a good idea to do this. Why not simply loop through the features?
    
    if option == 'file':
        driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    elif option == 'memory':
        driver = osgeo.ogr.GetDriverByName('Memory')
    layer_defn = input_layer.GetLayerDefn()
    # loop through the input features
    inFeature = input_layer.GetFeature(index)
    #if os.path.exists(output_shape):
        #driver.DeleteDataSource(output_shape) 
    outDS = driver.CreateDataSource(output_shape) #outDS = driver.CreateDataSource("out")

    field_names = [layer_defn.GetFieldDefn(j).GetName() for j in range(layer_defn.GetFieldCount())] #store the field names as a list of strings
        
    if outDS is None:
        print 'Could not create file'
        sys.exit(1)

    outLayer = outDS.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)

    # get the FieldDefn for the county name field
    for j in range(0,len(field_names)):
        field = inFeature.GetFieldDefnRef(field_names[j])
        outLayer.CreateField(field)

    # get the FeatureDefn for the output shapefile
    featureDefn = outLayer.GetLayerDefn()

    # get the input geometry
    geom = inFeature.GetGeometryRef()

    # create a new feature
    outFeature = osgeo.ogr.Feature(featureDefn)

    # set the geometry and attribute
    outFeature.SetGeometry(geom)
    for j in range(0,len(field_names)):
        outFeature.SetField(field_names[j],inFeature.GetField(field_names[j]))
    
    # add the feature to the shapefile
    outLayer.CreateFeature(outFeature)
    
    if option == 'memory':
        return outDS

    # destroy the features and get the next input feature
    outFeature.Destroy()
    inFeature.Destroy()
    outDS.Destroy()
    

def smooth_filter_otb(input_raster,output_raster,radius):
    
    '''Apply the Meanshift smoothing to the input image
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster (*.TIF,*.tiff) (string)
    :param radius: radius parameter (integer, 0 for default)
    :returns:  an output file is created according to the specified output mode
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Daniel Aurelio Galeazzo - Mostapha Harb
    Last modified: 20/05/2014
    
    Reference: http://www.orfeo-toolbox.org/CookBook/CookBooksu91.html#x122-5460005.5.2
    ''' 
    
    # The following line creates an instance of the MeanShiftSmoothing application 
    MeanShiftSmoothing = otbApplication.Registry.CreateApplication("MeanShiftSmoothing") 
    
    if radius == 0:
        radius = 30
    # The following lines set all the application parameters: 
    MeanShiftSmoothing.SetParameterString("in", input_raster) 
    MeanShiftSmoothing.SetParameterString("fout", output_raster)
    MeanShiftSmoothing.SetParameterInt("spatialr", radius)
    MeanShiftSmoothing.SetParameterFloat("ranger", radius) 
    MeanShiftSmoothing.SetParameterFloat("thres", 0.1) 
    MeanShiftSmoothing.SetParameterInt("maxiter", 100) 
     
    # The following line execute the application 
    MeanShiftSmoothing.ExecuteAndWriteOutput()


class Polygon(object):
    '''Container for geometry builders draw the minimun and maximum x and y values generated with WindowsMaker.make_coordinates method.

    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    ''' 

    class Square(object):
        '''Square Builder'''

        def __init__(self,x_min,x_max,y_min,y_max,resize=0):
            '''    
            :param x_min,x_max,y_min,y_max: coordinate of minimum and maximum x and y. (float)
            :param resize: resize value for increase dimension of the form expressed in coordinates as unit of measure (float)
            ''' 
            self.x_min,self.x_max,self.y_min,self.y_max,self.resize = x_min,x_max,y_min,y_max,resize

        def add(self):
            '''Creation of polygon

            :returns: poly: square polygon (ogr.Geometry)
            '''
            window = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
            window.AddPoint(self.x_min-self.resize, self.y_max+self.resize)
            window.AddPoint(self.x_max+self.resize, self.y_max+self.resize)
            window.AddPoint(self.x_max+self.resize, self.y_min-self.resize)
            window.AddPoint(self.x_min-self.resize, self.y_min-self.resize)
            window.CloseRings()
            poly = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
            poly.AddGeometry(window)
            return poly

    class Circle(Square):
        '''Circle Builder'''

        def add(self):
            '''Creation of polygon

            :returns: poly: circle polygon (ogr.Geometry)
            '''
            x_center = (self.x_max+self.x_min)/2
            y_center = (self.y_max+self.y_min)/2
            x_dist = math.ceil( math.sqrt( (self.y_max - self.y_min)**2 + (self.x_max - self.x_min)**2 ) ) + self.resize*2 + self.resize*2
            y_dist = math.ceil( math.sqrt( (self.y_max - self.y_min)**2 + (self.x_max - self.x_min)**2 ) ) + self.resize*2 + self.resize*2
            radius = max(x_dist,y_dist)/2
            point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
            point.AddPoint(x_center,y_center)
            circle = point.Buffer(radius,40)
            return circle


class WindowsMaker(object):
    ''' With this class it's possible to make a window around a ogr feature

    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''

    def __init__(self,inFeature,resize=0):
        '''
        :param inFeature: feature on whic make window around (ogr.Feature)
        :param resize: resize value for increase dimension of the form expressed in coordinates as unit of measure (float)
        '''
        self.inFeature = inFeature
        self.resize = resize

    def make_coordinates(self):
        '''Build coordinate of minimum and maximum x and y searching the minimum and maximum value of x and y into all point of feature

        :returns: self.x_min,self.x_max,self.y_min,self.y_max: coordinate of minimum and maximum x and y. (float)
        '''
        x_list = []
        y_list = []
        geom = self.inFeature.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        n_vertex = ring.GetPointCount()
        #Single polygon case
        if n_vertex:
            for i in range(n_vertex):
                lon,lat,z = ring.GetPoint(i)
                x_list.append(lon)
                y_list.append(lat)
            x_list.sort()
            self.x_min = x_list[0]
            self.x_max = x_list[len(x_list)-1]
            y_list.sort()
            self.y_min = y_list[0]
            self.y_max = y_list[len(y_list)-1]
        #Multipolygon case
        else:
            for i in range(ring.GetGeometryCount()+1):
                multigeom = geom.GetGeometryRef(i)
                ring = multigeom.GetGeometryRef(0)
                n_vertex = ring.GetPointCount()
                for c in range(n_vertex):
                    lon,lat,z = ring.GetPoint(i)
                    x_list.append(lon)
                    y_list.append(lat)
            x_list.sort()
            self.x_min = x_list[0]
            self.x_max = x_list[len(x_list)-1]
            y_list.sort()
            self.y_min = y_list[0]
            self.y_max = y_list[len(y_list)-1]
        return self.x_min,self.x_max,self.y_min,self.y_max
        
    def make_feature(self,geom=None,polygon=Polygon.Square):
        '''Build the window feauture

        :param geom: Geometry polygon of window (ogr.Geometry)
        :param polygon: Polygon builder (sensum.Polygon)
        :returns: feauture built (ogr.Feature)
        '''
        if geom == None:
            self.make_coordinates()
            geom = polygon(self.x_min,self.x_max,self.y_min,self.y_max,self.resize).add()
        driver = osgeo.ogr.GetDriverByName('Memory')
        ds = driver.CreateDataSource("")
        layer = ds.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)
        outFeature = osgeo.ogr.Feature(layer.GetLayerDefn())
        outFeature.SetGeometry(geom)
        self.windowFeature = outFeature
        return outFeature

    def get_shapeDS(self,inLayer):
        '''Build a layer with only intersected features of input layer

        :param inLayer: Layer which will be checked. (ogr.Layer)
        :returns: Dataset of new layer with only feature intersected. (ogr.Dataset)
        '''
        driver = osgeo.ogr.GetDriverByName('Memory')
        outDS = driver.CreateDataSource("ram")
        outLayer = outDS.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)
        n_features = inLayer.GetFeatureCount()
        for i in range(n_features):
            inFeature = inLayer.GetFeature(i)
            if self.windowFeature.GetGeometryRef().Intersect(inFeature.GetGeometryRef()):
                outLayer.CreateFeature(inFeature)
        return outDS

    def get_rasterArray(self,inRaster):
        '''Tile a raster with dimensions of window maked

        :param inRaster: Raster which will be tiled
        :returns: list of array represent the raster
        '''
        rows = inRaster.RasterYSize
        cols = inRaster.RasterXSize
        bands = inRaster.RasterCount

        transform = inRaster.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        xFirst = int(round((self.x_min - xOrigin) / pixelWidth))
        yFirst = int(round((self.y_max - yOrigin) / pixelHeight))
        xLast = int(round((self.x_max - xOrigin) / pixelWidth))
        yLast = int(round((self.y_min - yOrigin) / pixelHeight))

        #fix resize
        if xFirst < 0:
            xFirst = 0
        if yFirst < 0:
            yFirst = 0

        x_dimension = xLast - xFirst
        y_dimension = yLast - yFirst

        #fix resize
        if xFirst+x_dimension > cols:
            x_dimension = cols - xFirst
        if yFirst+y_dimension > rows:
            y_dimension = rows - yFirst
    
        for j in range(bands):
            band_list = inRaster.GetRasterBand(j+1).ReadAsArray(xFirst,yFirst,x_dimension,y_dimension)
        print band_list

    def make_shape(self,path):
        '''Shape file builder with only the window feature into the layer (useful for debugging)

        :param path: Path of shape file (str)
        '''
        driver = osgeo.ogr.GetDriverByName('Memory')
        ds = driver.CreateDataSource("")
        layer = ds.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)
        poly = self.get_geometry()
        outFeature = osgeo.ogr.Feature(layer.GetLayerDefn())
        outFeature.SetGeometry(poly)
        self.windowFeature = outFeature
        
        driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
        outDS = driver.CreateDataSource(path)
        outLayer = outDS.CreateLayer('polygon', geom_type=osgeo.ogr.wkbPolygon)
        outLayer.CreateFeature(outFeature)

def normalize_to_L8(input_band_list):
    
    '''Normalize the range of values of Landsat 5 and 7 to match Landsat 8
    
    :param input_band_list: list of 2darrays (list of 2darrays)
    :returns:  a list containing the normalized bands as ndarrays (list of arrays).
    :raises: AttributeError, KeyError
    
    Author: Mostapha Harb - Daniele De Vecchi - Daniel Aurelio Galeazzo
    Last modified: 24/06/2014
    
    '''
    out_list = []
    for b in range(0,len(input_band_list)):
        matrix = input_band_list[b].astype(np.float32)
        matrix = (matrix - matrix.min()) / (matrix.max()-matrix.min())
        matrix = matrix * 65000
        matrix = matrix.astype(np.int32)
        out_list.append(matrix)  
    
    return out_list 
    

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
