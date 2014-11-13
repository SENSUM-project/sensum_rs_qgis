'''
.. module:: preprocess
   :platform: Unix, Windows
   :synopsis: This module includes functions related to preprocessing of multi-spectral satellite images.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia
'''
'''
---------------------------------------------------------------------------------
                                preprocess.py
---------------------------------------------------------------------------------
Created on May 13, 2013
Last modified on Mar 19, 2014
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
import osgeo.gdal
from gdalconst import *
import cv2
import numpy as np
import osgeo.ogr
import otbApplication
import shutil
from conversion import *

if os.name == 'posix': 
    separator = '/'
else:
    separator = '\\'


def clip_rectangular(input_raster,data_type,input_shape,output_raster,mask=False,resize=0):
    
    '''Clip a raster with a rectangular shape based on the provided polygon
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :param input_shape: path and name of the input shapefile (*.shp) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param mask: bool to enable the usage of the shapefile as a mask (boolean)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    ''' 

    #TODO: Why not use gdalwarp?
    #TODO: would use only one argument to define input image and one to define input shp.
        
    #os.system('gdalwarp -q -cutline ' + shapefile + ' -crop_to_cutline -of GTiff ' + path + name +' '+ path + name[:-4] + '_city.TIF')
    #print input_raster
    if data_type == 0: data_type = np.uint16
    x_list = []
    y_list = []
    x_list_coordinates = []
    y_list_coordinates = []
    # get the shapefile driver
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    # open the data source
    datasource = driver.Open(input_shape, 0)
    if datasource is None:
        print 'Could not open shapefile'
        sys.exit(1)

    layer = datasource.GetLayer() #get the shapefile layer
    
    inb = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    if inb is None:
        print 'Could not open'
        sys.exit(1)
        
    geoMatrix = inb.GetGeoTransform()
    driver = inb.GetDriver()
    cols = inb.RasterXSize
    rows = inb.RasterYSize
    nbands = inb.RasterCount  
        
    # loop through the features in the layer
    feature = layer.GetNextFeature()
    while feature:
        try:
            # get the x,y coordinates for the point
            geom = feature.GetGeometryRef()
            ring = geom.GetGeometryRef(0)
            n_vertex = ring.GetPointCount()
            for i in range(0,n_vertex-1):
                lon,lat,z = ring.GetPoint(i)
                #x_matrix,y_matrix = world2pixel(geoMatrix,lon,lat)
                #x_list.append(x_matrix)
                #y_list.append(y_matrix)
                x_list_coordinates.append(lon)
                y_list_coordinates.append(lat)
            # destroy the feature and get a new one
            feature.Destroy()
            feature = layer.GetNextFeature()
        except:
            feature = None
    #regularize the shape
    
    x_list_coordinates.sort()
    x_min = x_list_coordinates[0]
    y_list_coordinates.sort()
    y_min = y_list_coordinates[0]
    x_list_coordinates.sort(None, None, True)
    x_max = x_list_coordinates[0]
    y_list_coordinates.sort(None, None, True)
    y_max = y_list_coordinates[0]
    

    #print x_min, geoMatrix[0]
    #print x_max, geoMatrix[0]+cols*geoMatrix[1]
    '''
    x_min = max(x_min,geoMatrix[0])
    y_min = min(y_min,geoMatrix[3]-rows*geoMatrix[1])
    x_max = min(x_max,geoMatrix[0]+cols*geoMatrix[1])
    y_max = max(y_max,geoMatrix[3])
    '''

    if x_min < geoMatrix[0]: x_min = geoMatrix[0]
    if y_min < geoMatrix[3]-rows*geoMatrix[1]: y_min = geoMatrix[3]-rows*geoMatrix[1]
    if x_max > geoMatrix[0]+cols*geoMatrix[1]: x_max = geoMatrix[0]+cols*geoMatrix[1]
    if y_max > geoMatrix[3]: y_max = geoMatrix[3]

    x_min = x_min - resize
    y_max = y_max + resize
    lon_min = x_min
    lat_min = y_max


    x_min, y_max = world2pixel(geoMatrix, x_min, y_max)
    x_max, y_min = world2pixel(geoMatrix, x_max, y_min)
    #compute the new starting coordinates
    #lon_min = float(x_min*geoMatrix[1]+geoMatrix[0]) 
    #lat_min = float(geoMatrix[3]+y_min*geoMatrix[5])

    geotransform = [lon_min,geoMatrix[1],0.0,lat_min,0.0,geoMatrix[5]]
    #print x_max,x_min
    #print y_max,y_min
    cols_out = x_max-x_min + resize
    rows_out = y_min-y_max + resize
    
    gdal_data_type = data_type2gdal_data_type(data_type)
    if mask == True:
        rows_ref,cols_ref,nbands_ref,geo_transform_ref,projection_ref = read_image_parameters(input_raster)
        shp2rast(input_shape,input_shape[:-4]+'.tif',rows_out,cols_out,'Mask',pixel_width=geo_transform_ref[1],pixel_height=abs(geo_transform_ref[5]),x_min=0,x_max=0,y_min=0,y_max=0) 
        mask_list = read_image(input_shape[:-4]+'.tif',np.uint8,0)
        msk = np.equal(mask_list[0],1)
    output=driver.Create(output_raster,cols_out,rows_out,nbands,gdal_data_type) #to check
    
    for b in range (1,nbands+1):
        inband = inb.GetRasterBand(b)
        data = inband.ReadAsArray(x_min,y_max,cols_out,rows_out).astype(data_type)
        if mask == True:
            data = np.choose(msk,(0,data))
        outband=output.GetRasterBand(b)
        outband.WriteArray(data,0,0) #write to output image
    
    output.SetGeoTransform(geotransform) #set the transformation
    output.SetProjection(inb.GetProjection())
    # close the data source and text file
    datasource.Destroy()
    

def layer_stack(input_raster_list,output_raster,data_type):
    
    '''Merge single-band files into one multi-band file
    
    :param input_raster_list: list with paths and names of the input raster files (*.TIF,*.tiff) (list of strings)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    ''' 
    
    final_list = []
    for f in range(0,len(input_raster_list)): #read image by image
        band_list = read_image(input_raster_list[f],data_type,0)
        rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster_list[f])
        final_list.append(band_list[0]) #append every band to a unique list
        
    write_image(final_list,data_type,0,output_raster,rows,cols,geo_transform,projection) #write the list to output file
    
    
def layer_split(input_raster,band_selection,data_type):
    
    '''Split a multi-band input file into single-band files
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param band_selection: number associated with the band to extract (0: all bands, 1: blue, 2: greeen, 3:red, 4:infrared) (integer)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :returns:  an output file is created for single-band
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 18/03/2014
    '''

    #TODO: Do we need this?
    #TODO: Would rename arguments merge(src_img, dst_dir, option)
    
    band_list = read_image(input_raster,data_type,band_selection)
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    if band_selection == 0:
        for b in range(1,nbands+1):
            write_image(band_list,data_type,b,input_raster[:-4]+'_B'+str(b)+'.TIF',rows,cols,geo_transform,projection)
    else:
        write_image(band_list,data_type,band_selection,input_raster[:-4]+'_B'+str(band_selection)+'.TIF',rows,cols,geo_transform,projection)  
    

def gcp_extraction(input_band_ref,input_band,ref_geo_transform,output_option):
    
    '''GCP extraction and filtering using the SURF algorithm
    
    :param input_band_ref: 2darray byte format (numpy array) (unsigned integer 8bit)
    :param input_band: 2darray byte format (numpy array) (unsigned integer 8bit)
    :param ref_geo_transform: geomatrix related to the reference image
    :param output_option: 0 for indexes, 1 for coordinates (default 0) (integer)
    :param data_type: numpy type used to read the image (e.g. np.uint8, np.int32; 0 for default: np.uint16) (numpy type)
    :returns:  an output file is created for single-band
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    #TODO: It takes only a 2d array (so only one image band) and not the full image content?
    #TODO: 2d array is created by using Read_Image() -> band_list[i]?
    #TODO: So we have two type of functions: 1. functions that take directly a file (e.g. geotiff) and 2. functions that take an array?
    #TODO: Would rename function to something like auto_gcp()
    #TODO: Output a list of gcps following the structure required by gdal_transform -> this way we could use gdal for the actual transformation and only focus on a robuts and flexible gcp detection
    #TODO: We should think of an option to manually adjust auto gcps for example using QGIS georeferencer (comment from Dilkushi during skype call 7.3.2014)
    #C:\OSGeo4W\bin
    detector = cv2.FeatureDetector_create("SURF") 
    descriptor = cv2.DescriptorExtractor_create("BRIEF")
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
    # detect keypoints
    kp1 = detector.detect(input_band_ref)
    kp2 = detector.detect(input_band)
    
    # descriptors
    k1, d1 = descriptor.compute(input_band_ref, kp1)
    k2, d2 = descriptor.compute(input_band, kp2)
    
    # match the keypoints
    matches = matcher.match(d1, d2)
    
    # visualize the matches
    dist = [m.distance for m in matches] #extract the distances
    a=sorted(dist) #order the distances
    fildist=np.zeros(1) #use 1 in order to select the most reliable matches
    
    for i in range(0,1):
        fildist[i]=a[i]
    thres_dist = max(fildist)
    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance <= thres_dist] 
    
    i=0
    points=np.zeros(shape=(len(sel_matches),4))
    points_coordinates = np.zeros(shape=(len(sel_matches),4)).astype(np.float32)
    for m in sel_matches:
        #matrix containing coordinates of the matching points
        points[i][:]= [int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1]),int(k2[m.trainIdx].pt[0]),int(k2[m.trainIdx].pt[1])]
        i=i+1
    #include new filter, slope filter not good for rotation
    #include conversion from indexes to coordinates
    #print 'Feature Extraction - Done'
    if output_option == None or output_option == 0:
        return points #return indexes
    else: #conversion to coordinates
        for j in range(0,len(points)):
            lon_ref,lat_ref = pixel2world(ref_geo_transform, points[j][0], points[j][1])
            lon_tg,lat_tg = pixel2world(ref_geo_transform, points[j][2], points[j][3]) #check how the gdal correction function works
            points_coordinates[j][:] = [lon_ref,lat_ref,lon_tg,lat_tg]
        return points_coordinates 


def linear_offset_comp(common_points):
    
    '''Linear offset computation using points extracted by gcp_extraction
    
    :param common_points: matrix with common points extracted by gcp_extraction (matrix of integers)
    :returns:  list with x and y offset
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    xoff1=np.zeros(len(common_points)) 
    yoff1=np.zeros(len(common_points))
    
    #Offset calculation band1
    for l in range(0,len(common_points)):
        xoff1[l]=common_points[l][2]-common_points[l][0]
        yoff1[l]=common_points[l][3]-common_points[l][1]
   
    #Final offset calculation - mean of calculated offsets
    xoff=round((xoff1.mean())) #mean computed in case of more than one common point
    yoff=round((yoff1.mean())) #mean computed in case of more than one common point
    
    return xoff,yoff
        

def pansharp(input_raster_multiband,input_raster_panchromatic,output_raster):
    
    '''Pansharpening operation using OTB library
    
    :param input_raster_multiband: path and name of the input raster multi-band file (*.TIF,*.tiff) (string)
    :param input_raster_panchromatic: path and name of the input raster panchromatic file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb - Daniel Aurelio Galeazzo
    Last modified: 23/05/2014
    '''

    #TODO: Specify in description which pansharpening algorithm iss used by this function
    fix_tiling_raster(input_raster_multiband,input_raster_panchromatic)
    rowsp,colsp,nbands,geo_transform,projection = read_image_parameters(input_raster_panchromatic)
    rowsxs,colsxs,nbands,geo_transform,projection = read_image_parameters(input_raster_multiband)
 
    scale_rows = round(float(rowsp)/float(rowsxs),6)
    scale_cols = round(float(colsp)/float(colsxs),6)
    print scale_rows,scale_cols
    #Resampling
    RigidTransformResample = otbApplication.Registry.CreateApplication("RigidTransformResample") 
    # The following lines set all the application parameters: 
    RigidTransformResample.SetParameterString("in", input_raster_multiband) 
    RigidTransformResample.SetParameterString("out", input_raster_multiband[:-4]+'_resampled.tif') 
    RigidTransformResample.SetParameterString("transform.type","id") 
    RigidTransformResample.SetParameterFloat("transform.type.id.scalex", scale_cols) 
    RigidTransformResample.SetParameterFloat("transform.type.id.scaley", scale_rows) 
    RigidTransformResample.SetParameterInt("ram", 2000)
    RigidTransformResample.ExecuteAndWriteOutput()
 
    Pansharpening = otbApplication.Registry.CreateApplication("Pansharpening") 
    # Application parameters
    Pansharpening.SetParameterString("inp", input_raster_panchromatic) 
    Pansharpening.SetParameterString("inxs", input_raster_multiband[:-4]+'_resampled.tif') 
    Pansharpening.SetParameterInt("ram", 2000) 
    Pansharpening.SetParameterString("out", output_raster) 
    Pansharpening.SetParameterOutputImagePixelType("out", 3) 
     
    Pansharpening.ExecuteAndWriteOutput()
    
    
def resampling(input_raster,output_raster,output_resolution,resampling_algorithm):
    
    '''Resampling operation using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param output_resolution: resolution of the outout raster file (float)
    :param resampling_algorithm: choice among different algorithms (nearest_neigh,linear,bicubic)
    :returns:  an output file is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 19/03/2014
    '''
    
    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_raster)
    scale_value = round(float(geo_transform[1])/float(output_resolution),4)
    RigidTransformResample = otbApplication.Registry.CreateApplication("RigidTransformResample") 
    # The following lines set all the application parameters: 
    RigidTransformResample.SetParameterString("in", input_raster) 
    RigidTransformResample.SetParameterString("out", output_raster) 
    RigidTransformResample.SetParameterString("transform.type","id") 
    RigidTransformResample.SetParameterFloat("transform.type.id.scalex", scale_value) 
    RigidTransformResample.SetParameterFloat("transform.type.id.scaley", scale_value) 
    
    if resampling_algorithm == 'nearest_neigh': 
        RigidTransformResample.SetParameterString("interpolator","nn")
    if resampling_algorithm == 'linear':
        RigidTransformResample.SetParameterString("interpolator","linear")
    if resampling_algorithm == 'bicubic':
        RigidTransformResample.SetParameterString("interpolator","bco")
    
    RigidTransformResample.SetParameterInt("ram", 2000) 
    
    RigidTransformResample.ExecuteAndWriteOutput()
    
    
def fix_tiling_raster(input_raster1,input_raster2):
    '''Fix two images dimension for pansharpening issue instroducted by otb 4.0 version (seem to be otb 4.0 bug)

    :param input_raster1: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param input_raster1: path and name of the input raster file (*.TIF,*.tiff) (string)
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''
    minx1,miny1,maxx1,maxy1 = get_coordinate_limit(input_raster1)
    minx2,miny2,maxx2,maxy2 = get_coordinate_limit(input_raster2)
    os.system("gdal_translate -of GTiff -projwin "+str(minx1)+" "+str(maxy1)+" "+str(maxx1)+" "+str(miny1)+" "+input_raster2+" "+input_raster2+"_tmp.tif")
    shutil.move(input_raster2+"_tmp.tif",input_raster2)


def get_coordinate_limit(input_raster):
    '''Get corner cordinate from a raster

    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :returs: minx,miny,maxx,maxy: points taken from geomatrix (string)
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''
    dataset = osgeo.gdal.Open(input_raster, GA_ReadOnly)
    if dataset is None:
        print 'Could not open'
        sys.exit(1)
    driver = dataset.GetDriver()
    band = dataset.GetRasterBand(1)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geoMatrix = dataset.GetGeoTransform()
    minx = geoMatrix[0]
    miny = geoMatrix[3] + width*geoMatrix[4] + height*geoMatrix[5] 
    maxx = geoMatrix[0] + width*geoMatrix[1] + height*geoMatrix[2]
    maxy = geoMatrix[3]

    return minx,miny,maxx,maxy


def Extraction(image1,image2):
    
    '''
    ###################################################################################################################
    Feature Extraction using the SURF algorithm
    
    Input:
     - image1: path to the reference image - each following image is going to be matched with this reference
     - image2: path to the image to be corrected
    
    Output:
    Returns a matrix with x,y coordinates of matching points and the minimum distance
    ###################################################################################################################
    '''
    image_1 = osgeo.gdal.Open(image1,GA_ReadOnly)
    inband1=image_1.GetRasterBand(1)
    cols = image_1.RasterXSize
    img1 = inband1.ReadAsArray().astype('uint8')

    img1_m = np.ma.masked_values(img1, 0).astype('uint8')

    detector = cv2.FeatureDetector_create("SURF") 
    descriptor = cv2.DescriptorExtractor_create("BRIEF")
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    #print img1.type

    kp1 = detector.detect(img1, mask=img1_m)
    print img1_m
    k1, d1 = descriptor.compute(img1, kp1)

    image_1 = None
    img1 = None
    
    image_2 = osgeo.gdal.Open(image2,GA_ReadOnly)
    inband2=image_2.GetRasterBand(1)
    img2 = inband2.ReadAsArray().astype('uint8')

    img2_m = np.ma.masked_values(img2, 0).astype('uint8')
    
    kp2 = detector.detect(img2, mask = img2_m)
    k2, d2 = descriptor.compute(img2, kp2)

    image_2 = None
    img2 = None

    # match the keypoints
    matches = matcher.match(d1, d2)
    
    # visualize the matches
    dist = [m.distance for m in matches] #extract the distances
  
    thres_dist = 100
    sel_matches = [m for m in matches if m.distance <= thres_dist]

    points=np.zeros(shape=(len(sel_matches),4))
    points_shift = np.zeros(shape=(len(sel_matches),2))
    points_shift_abs = np.zeros(shape=(len(sel_matches),1))

    # Creo una variabile dove vado a scrivere, per ogni coppia: distanza hamming, shift, pendenza
   
    compar_stack = np.array([100,1.5,0.0,1,1,2,2])

    #vishualization(img1,img2,sel_matches)
    i = 0
    for m in sel_matches:
        points[i][:]= [int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1]),int(k2[m.trainIdx].pt[0]),int(k2[m.trainIdx].pt[1])]
        points_shift[i][:] = [int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]),int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1])]
        points_shift_abs [i][:] = [np.sqrt((int(cols + k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))**2+
                                           (int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))**2)]
        
        #print m.distance,'   ' , [np.sqrt((int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))**2+
                                          # (int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))**2)]
        
        deltax = np.float(int(k2[m.trainIdx].pt[0])-int(k1[m.queryIdx].pt[0]))
        deltay = np.float(int(k2[m.trainIdx].pt[1])-int(k1[m.queryIdx].pt[1]))
        
        if deltax == 0 and deltay != 0:
            slope = 90
        elif deltax == 0 and deltay == 0:
            slope = 0
        else:
            slope = (np.arctan(deltay/deltax)*360)/(2*np.pi)
        
        compar_stack = np.vstack([compar_stack,[m.distance,points_shift_abs [i][:],slope,
                                                int(k1[m.queryIdx].pt[0]),
                                                int(k1[m.queryIdx].pt[1]),
                                                int(k2[m.trainIdx].pt[0]),
                                                int(k2[m.trainIdx].pt[1])]])
        i=i+1

        
        
    #Ordino lo stack
    compar_stack = compar_stack[compar_stack[:,0].argsort()]#Returns the indices that would sort an array.
    #print compar_stack#[0:30]
    print len(compar_stack)

    best = best_row_2(compar_stack[0:90])#The number of sorted points to be passed
    print 'BEST!!!!!', best
        
    report = os.path.join(os.path.dirname(image1),'report.txt')
    out_file = open(report,"a")
    out_file.write("\n")
    out_file.write("Il migliore ha un reliability value pari a "+str(best[0])+" \n")
    out_file.close()
    migliore = [best[3:7]]
    #return migliore,best[0]
    return migliore

def best_row_2(compstack):
    '''
    ###################################################################################################################
    
    Input:
     - compstack: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    # Sort
    compstack = compstack[compstack[:,2].argsort()]
    spl_slope = np.append(np.where(np.diff(compstack[:,2])>0.1)[0]+1,len(compstack[:,0]))
    
    step = 0
    best_variability = 5
    len_bestvariab = 0
    best_row = np.array([100,1.5,0.0,1,1,2,2])

    for i in spl_slope:
        slope = compstack[step:i][:,2]
        temp = compstack[step:i][:,1]
        variab_temp = np.var(temp)
        count_list=[]
        if variab_temp <= best_variability and len(temp) >3:
            count_list.append(len(temp))

            if variab_temp < best_variability:
                
                best_variability = variab_temp
                len_bestvariab = len(temp)                
                best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                all_rows = compstack[step:i]
            if variab_temp == best_variability:
                if len(temp)>len_bestvariab:
                    best_variability = variab_temp
                    len_bestvariab = len(temp)                
                    best_row = compstack[step:i][compstack[step:i][:,0].argsort()][0]
                    all_rows = compstack[step:i]
        step = i
    return best_row#,,point_list1,point_list2



def affine_corrected(img,delta_x,delta_y):
    '''
    ###################################################################################################################
    
    Input:
     - img,delta_x,delta_y: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    rows,cols = img.shape  
    M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


def F_B(path,floating,ref):
    '''
    ###################################################################################################################
    
    Input:
     - path,floating,ref: ..........................
    
    Output:
    ..............................
    ###################################################################################################################
    '''
    dir1 = os.listdir(floating)
    dir2 = os.listdir(ref)
    a_list=[]
    for i in ban:#range(1,6):#(1,6):
        ref_list = [s for s in dir1 if "_B"+str(i)+'_city' in s ]
        if len(ref_list):
            floating_list = [s for s in dir2 if "_B"+str(i)+'_city' in s ]
            rows,cols_q,nbands,geotransform,projection = read_image_parameters(floating+ref_list[0])
            rows,cols_q,nbands,geotransform,projection = read_image_parameters(ref+floating_list[0])
            band_list0 = read_image(floating+ref_list[0],np.uint8,0)
            band_list1 = read_image(ref+floating_list[0],np.uint8,0)
            im0=band_list0[0]
            im1=band_list1[0]       
            a=Extraction(floating+ref_list[0],ref+floating_list[0])# the coordinates of the max point which is supposed to be the invariant point
            a_list.append(a[0][0] - a[0][2])
            a_list.append(a[0][1] - a[0][3])
            b=affine_corrected(im0,a[0][2] - a[0][0],a[0][3] - a[0][1])
            out_list=[]
            out_list.append(b)
            write_image(out_list,0,i,path+'corrected_Transl._'+floating[-11:-1]+'_B'+str(i)+'.tif',rows,cols_q,geotransform,projection)       
    return a_list
    