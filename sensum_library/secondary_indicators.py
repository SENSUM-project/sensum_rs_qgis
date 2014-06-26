'''
.. module:: secondary_indicators
   :platform: Unix, Windows
   :synopsis: This module includes functions related to the high-level classification of multi-spectral satellite images.

.. moduleauthor:: Mostapha Harb <mostapha.harb@eucentre.it>
.. moduleauthor:: Daniele De Vecchi <daniele.devecchi03@universitadipavia.it>
.. moduleauthor:: Daniel Aurelio Galeazzo <dgaleazzo@gmail.com>
   :organization: EUCENTRE Foundation / University of Pavia 
'''
'''
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
import osgeo.gdal, gdal
from gdalconst import *
import numpy as np
import otbApplication
from skimage.segmentation import felzenszwalb, slic, quickshift
from scipy import optimize
from scipy import ndimage
import shutil
import glob
import collections
import ephem
import math
from operator import itemgetter
import operator
from collections import defaultdict,Counter
import osr
import osgeo.ogr, ogr
from conversion import *

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'
    
    
def shadow_length(input_band,latitude,longitude,date):
    
    '''Compute the shadow length using the angle from the sun position
    
    :param input_band: 2darray with the extracted shadow
    :param latitude: decimal latitude (float)
    :param longitude: decimal longitude (float)
    :param date: acquisition date of the image (yyyy/mm/dd) (string)
    :returns:  length of the shadow in pixels (float)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    
    Reference: http://rhodesmill.org/pyephem/quick.html
    ''' 
    
    #TODO: Problem with Sun() function in ephem.
    #TODO: which kind of raster input is required? not clear from description.
    #TODO: length of shadow from were to were? Building location information needed?
    
    o = ephem.Observer()
    o.lat, o.long,o.date = latitude,longitude,date
    #print 'o.lat,o.long',o.lat,o.long
    sun = ephem.Sun(o) #not an error
    azimuth = sun.az
    angle= math.degrees(azimuth)         
    rot = ndimage.interpolation.rotate(input_band, angle)
    #print 'azimuth_angle',angle
    c=np.apply_along_axis(sum,1, rot)
    return max(c)


def building_height(latitude,longitude,date,shadow_len):
    
    '''Compute the building height using the angle from the sun position and the shadow length
    
    :param latitude: decimal latitude (float)
    :param longitude: decimal longitude (float)
    :param date: acquisition date of the image (yyyy/mm/dd) (string)
    :param shadow_len: length of the shadow computed with the function
    :returns:  height of the building in pixels (float)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    ''' 
    
    o = ephem.Observer()
    o.lat, o.long, o.date = latitude,longitude,date
    sun = ephem.Sun(o) 
    A = sun.alt
    building_height = math.tan(A)*shadow_len
    azimuth = sun.az
    azimuth= math.degrees(azimuth)
    building_height=round(building_height, 2)
    return building_height


def building_alignment(input_band):
    
    '''Compute the building alignment
    
    :param input_band: 2darray with the extracted building
    :returns:  alignment of the building in degrees (resolution of 15 degrees) and length along the 2 axis (alignment,length_a,length_b) (list of integers)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    ''' 

    #TODO: What is the input?, Would need an explanation of the following functions.
 
    angle=[0,15,30,45,60,75,90,105,120,135,150,165]  
    max_freq_c = 0
    max_freq_r = 0
    alpha = 0
    #looping on the angles
    for i in range(0,len(angle)):    
        
        rot = ndimage.interpolation.rotate(input_band, angle[i])
        
        c = np.sum(rot,axis=0)
        r = np.sum(rot,axis=1)
        #print c
        x1_c = Counter(c)
        x1_r = Counter(r)
        #p1=[elt1 for elt1,count1 in x1.most_common()]
        y1_c = (x1_c.most_common())
        y1_r = (x1_r.most_common())
        #possible values are 0,1,10,11
        y2_c = sorted(y1_c,key=itemgetter(1), reverse=True) #take the most frequent value
        y2_r = sorted(y1_r,key=itemgetter(1), reverse=True)
        
        if y2_c[0][1] > max_freq_c:
            max_freq_c = y2_c[0][1]
            #max_freq_r = y2_r[0][1]
            #alpha = 90+angle[i]
            alpha = 180-angle[i]
            y3_c = sorted(y1_c,key=itemgetter(0), reverse=True)
            y3_r = sorted(y1_r,key=itemgetter(0), reverse=True)
            length_a = y3_c[0][0]
            length_b = y3_r[0][0]

    if alpha == 180:
        alpha = 0
    print alpha

    return alpha,length_a,length_b
    

def building_regularity(length_a,length_b):
    
    '''Compute the building regularity (still missing the L-shape factor)
    
    :param length_a: length along one axis (integer)
    :param length_b: length along the other axis (integer)
    :returns:  'regular' or 'irregular' according to the regularity index (string)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 25/03/2014
    '''          
   
    if length_a > length_b:
        reg_ind = length_a / length_b
    else:
        reg_ind = length_b / length_a
    
    if 0<=reg_ind<=4:
        return 'regular'
    elif reg_ind>4 :
        return 'irregular'
    elif reg_ind<0:
        raise Exception("wrong irregularity index")

class ShadowPosition:

    '''Class for determinate the position of shadow than building calculated with date of raster acquisition \
    and coordinates.
    '''

    def __init__(self, date, latitude=None, longitude=None, rasterpath=None):

        '''Get shadow position compared with a feature

        latitude/longitude or rasterpath are mandatories for azimuth calculation

        :param date: date of acquisition in EDT (example: '1984/5/30 17:40:56') (char)
        :param latitude: Latitude (example: '33.775867') (char/int/float)
        :param longitude: Longitude (example: '-84.39733') (char/int/float)
        :param rasterpath: Path of raster for coordinate extraction
        '''
        self.longitude, self.latitude, self.date = longitude, latitude, date
        if rasterpath:
            self.set_raster(rasterpath)

    #TODO implement more elegant pattern
    def main(self):

        '''Commander method

        :returns: operator functions (tuple of 2 function)
        '''
        self.get_azimuth()
        self.azirange()
        self.leftOrRight, self.upOrDown = self.azirange()
        return self.leftOrRight, self.upOrDown

    def get_azimuth(self):

        '''Calculate the azimuth with coordinates
        '''
        try:
            gatech = ephem.Observer()
            gatech.lat = str(self.latitude)
            gatech.lon = str(self.longitude)
            gatech.date = str(self.date)
            v = ephem.Sun(gatech)
            self.azimuth = math.degrees(v.az)
            self.alt = v.alt
            if math.degrees(self.alt) < 0:
                print "Warning: No sun at {} EDT in this cordinates: {},{}. Angle is: {}".format(self.date,self.latitude,self.longitude,self.alt)
            elif math.degrees(self.alt) > 80:
                print "Warning: Sun is near to be perpendicular to the plane that mean shadow might be not fine defined or completely not visible at {} EDT in this cordinates: {},{}. Angle is: {}".format(self.date,self.latitude,self.longitude,v.alt)
                
        except TypeError:
            print "TypeError: Need to set longitude and latitude through __init__(), set_coordinates() or set set_raster()"

    #TODO implement same thing with shapefiles
    def set_raster(self,rasterpath):

        '''Get coordinate from a raster

        :param rasterpath: path of raster (str)
        '''
        dataset = gdal.Open(rasterpath, GA_ReadOnly)
        geo_transform = dataset.GetGeoTransform()
        prj = dataset.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        zone = int(str(srs.GetAttrValue('projcs')).split('zone ')[1][:-1])
        easting = geo_transform[0]
        northing = geo_transform[3]
        a = utm2wgs84(easting, northing, zone)
        self.latitude,self.longitude = a[0],a[1]

    def azirange(self):

        '''Get position of shadow than feature

        :returns: position of shadow than building (tuple of 2 str)
        '''
        if self.azimuth > 355 or self.azimuth < 5:
            return "CENTRE","DOWN"
        else:
            aziranges = [[5,85,"LEFT", "DOWN"],[85,95,"LEFT","CENTRE"],[95,175,"LEFT","UP"],[175,185,"CENTRE","UP"],[185,265,"RIGHT","UP"],[265,275,"RIGHT","CENTRE"],[275,355,"RIGHT","DOWN"]]
            return ((azirange[2],azirange[3]) for azirange in aziranges if self.azimuth >= azirange[0] and self.azimuth <= azirange[1]).next()

    #TODO thinking better implementation, this method is not logically related with class
    def operator(self):

        '''Get operator for if statemant of WindowsMaker class

        :returns: operator functions (tuple of 2 functions)
        '''

        '''
        | lt: < | le: <= | eq: == | gt: > | ge: >= |
        using sum operator for have value always > 0 that mean that when shadow    \
        is in CENTRE position will ignore the relative coordinate checking         \
        '''

        if self.leftOrRight == 'RIGHT' and self.upOrDown == 'UP':
            xOperator = operator.gt
            yOperator = operator.gt
        if self.leftOrRight == 'RIGHT' and self.upOrDown == 'CENTRE':
            xOperator = operator.gt
            yOperator = operator.add
        if self.leftOrRight == 'RIGHT' and self.upOrDown == 'DOWN':
            xOperator = operator.gt
            yOperator = operator.lt
        if self.leftOrRight == 'CENTRE' and self.upOrDown == 'UP':
            xOperator = operator.add
            yOperator = operator.gt
        if self.leftOrRight == 'CENTRE' and self.upOrDown == 'DOWN':
            xOperator = operator.add
            yOperator = operator.lt
        if self.leftOrRight == 'LEFT' and self.upOrDown == 'UP':
            xOperator = operator.lt
            yOperator = operator.gt
        if self.leftOrRight == 'LEFT' and self.upOrDown == 'CENTRE':
            xOperator = operator.lt
            yOperator = operator.add
        if self.leftOrRight == 'LEFT' and self.upOrDown == 'DOWN':
            xOperator = operator.lt
            yOperator = operator.lt
        return xOperator, yOperator


def shadow_checker(buildingShape, shadowShape, date, idfield="ID", outputShape='', resize=0):

    '''Function to assign right shadows to buildings using the azimuth calculated from the raster acquisition date and time.\
    Build new dataset (or shapefile if outputShape argument declared) with field ShadowID and Height related to   \
    the id of relative shadow fileshape and the height of building calculated with sensum.ShadowPosition \
    class. Polygons are taken from the building-related layer. Shadow shapefile has to be just processed with height() function. 

    :param buildingShape: path of buildings shapefile (str)
    :param shadowShape: path of shadows shapefile (str)
    :param date: date of acquisition in EDT (example: '1984/5/30 17:40:56') (char)
    :param idfield: id field of shadows (char)
    :param outputShape: path of output shapefile (char)
    :param resize: resize value for increase dimension of the form expressed in coordinates as unit of measure (float)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    #get layers
    driver = ogr.GetDriverByName("ESRI Shapefile")
    buildingsDS = driver.Open(buildingShape)
    buildingsLayer = buildingsDS.GetLayer()
    buindingsFeaturesCount = buildingsLayer.GetFeatureCount()
    shadowDS = driver.Open(shadowShape)
    shadowLayer = shadowDS.GetLayer()
    if outputShape == '':
        driver = ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outputShape)
    #copy the buildings layer and use it as output layer
    outDS.CopyLayer(buildingsLayer,"")
    outLayer = outDS.GetLayer()
    #add fields 'ShadowID' and 'Height' to outLayer
    fldDef = ogr.FieldDefn('ShadowID', ogr.OFTInteger)
    outLayer.CreateField(fldDef)
    fldDef = ogr.FieldDefn('Height', ogr.OFTReal)
    outLayer.CreateField(fldDef)
    #Calculate zone
    spatialRef = outLayer.GetSpatialRef()
    zone = int(str(spatialRef.GetAttrValue('projcs')).split('Zone')[1][1:-1])
    #get latitude and longitude from 1st features for calculate azimuth
    buildingFeature = outLayer.GetNextFeature()
    buildingGeometry = buildingFeature.GetGeometryRef().Centroid()
    longitude = float(buildingGeometry.GetX())
    latitude = float(buildingGeometry.GetY())
    longitude, latitude, altitude = utm2wgs84(longitude, latitude, zone)
    #calculate azimut and get right operators (> and <) as function used for filter the position of shadow
    position = ShadowPosition(date,latitude=latitude,longitude=longitude)
    position.main()
    xOperator, yOperator = position.operator()
    #loop into building features goint to make a window around each features and taking only shadow features there are into the window. 
    for i in range(buindingsFeaturesCount):
        print "{} di {} features".format(i+1,buindingsFeaturesCount)
        buildingFeature = outLayer.GetFeature(i)
        #make a spatial layer and get the layer
        maker = WindowsMaker(buildingFeature,resize)
        maker.make_feature()
        spatialDS = maker.get_shapeDS(shadowLayer)
        spatialLayer = spatialDS.GetLayer()
        spatialLayerFeatureCount = spatialLayer.GetFeatureCount()
        #check if there are any features into the spatial layer
        if spatialLayerFeatureCount:
            spatialFeature = spatialLayer.GetNextFeature()
            outFeature = outLayer.GetFeature(i)
            outGeometry = outFeature.GetGeometryRef().Centroid()
            xOutGeometry = float(outGeometry.GetX())
            yOutGeometry = float(outGeometry.GetY())
            #if the count of features is > 1 need to find the less distant feature
            if spatialLayerFeatureCount > 1:
                fields = []
                centros = []
                heights = []
                #loop into spatial layer
                while spatialFeature:
                    #calc of centroid of spatial feature
                    spatialGeometry = spatialFeature.GetGeometryRef().Centroid()
                    xSpatialGeometry = float(spatialGeometry.GetX())
                    ySpatialGeometry = float(spatialGeometry.GetY())
                    #filter the position of shadow, need to respect the right azimuth
                    if xOperator(xSpatialGeometry,xOutGeometry) and yOperator(ySpatialGeometry,yOutGeometry):
                        #saving id,centroid and heights into lists
                        fields.append(spatialFeature.GetField(idfield))
                        centros.append(spatialFeature.GetGeometryRef().Centroid())
                        heights.append(spatialFeature.GetField('Height'))
                    spatialFeature = spatialLayer.GetNextFeature()
                #calc distance from centroids
                distances = [outGeometry.Distance(centro) for centro in centros]
                #make multilist named datas with: datas[0] = distances, datas[1] = fields, datas[2] = heights and order for distances
                datas = sorted(zip(distances,fields,heights))
                #take less distant
                if datas:
                    outFeature.SetField("ShadowID",datas[0][1])
                    outFeature.SetField("Height",datas[0][2])
                    outLayer.SetFeature(outFeature)
            #spatialFeature = 1 so don't need to check less distant
            else:
                spatialGeometry = spatialFeature.GetGeometryRef().Centroid()
                xSpatialGeometry = float(spatialGeometry.GetX())
                ySpatialGeometry = float(spatialGeometry.GetY())
                if xOperator(xSpatialGeometry,xOutGeometry) and yOperator(ySpatialGeometry,yOutGeometry):
                    field = spatialFeature.GetField(idfield)
                    outFeature.SetField("ShadowID",field)
                    field = spatialFeature.GetField("Height")
                    outFeature.SetField("Height",field)
                    outLayer.SetFeature(outFeature)
                    spatialFeature = spatialLayer.GetNextFeature()
    return outDS

class CircleDensity(object):

    '''Circle Builder, circle maked with cendroid point
    '''

    def __init__(self,centroid,radius):
        
        '''
        :param centroid: centroid point (ogr.wkbPoint)
        :param radius: radius of circle expressed in coordinates as unit of measure (float)
        '''
        self.centroid = centroid
        self.radius = radius
        
    def add(self):

        '''
        :returns: geometry of polygon (ogr.Geometry)
        '''
        circle = self.centroid.Buffer(self.radius,40)
        return circle

def density(buildingShape,radius,outputShape=''):

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
    driver = ogr.GetDriverByName("ESRI Shapefile")
    buildingsDS = driver.Open(buildingShape)
    buildingsLayer = buildingsDS.GetLayer()
    buindingsFeaturesCount = buildingsLayer.GetFeatureCount()
    if outputShape == '':
        driver = ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outputShape)
    #copy the buildings layer and use it as output layer
    outDS.CopyLayer(buildingsLayer,"")
    outLayer = outDS.GetLayer()
    #add fields 'Density' to outLayer
    fldDef = ogr.FieldDefn('N_Building', ogr.OFTInteger)
    outLayer.CreateField(fldDef)
    fldDef = ogr.FieldDefn('Density', ogr.OFTReal)
    outLayer.CreateField(fldDef)
    #loop into building features goint to make a window around each features and taking how many buildings are there around
    for i in range(buindingsFeaturesCount):
        print "{} di {} features".format(i+1,buindingsFeaturesCount)
        buildingFeature = outLayer.GetFeature(i)
        #make a spatial layer and get the layer
        maker = WindowsMaker(buildingFeature)
        centroid = buildingFeature.GetGeometryRef().Centroid()
        maker.make_feature(geom=CircleDensity(centroid,radius).add())
        area = radius**2 * math.pi
        spatialDS = maker.get_shapeDS(buildingsLayer)
        spatialLayer = spatialDS.GetLayer()
        spatialLayerFeatureCount = spatialLayer.GetFeatureCount() -1 #(-1) for remove itself
        outFeature = outLayer.GetFeature(i)
        outFeature.SetField("N_Building",spatialLayerFeatureCount)
        if spatialLayerFeatureCount:
            outFeature.SetField("Density",float(spatialLayerFeatureCount/area))
        else:
            outFeature.SetField("Density",0)
        outLayer.SetFeature(outFeature)
    return outDS

#TODO need to check if pixelWidth and pixelHeight are necessary since if we know the coordinate system from shape we know the value of dimension in meter
def height(shadowShape,pixelWidth,pixelHeight,outShape=''):
    
    '''Function for calculate and assign to shadows shapefile the height of building and length of shadow.            \
    Build new dataset (or shapefile if outputShape argument declared) with field Shadow_Len and Height which contains \
    respectively the height of relative building and the length of the shadow                                         \
    
    :param shadowShape: path of shadows shapefile (str)
    :param pixelWidth: value in meters of length of 1 pixel (float)
    :param pixelHeight: value in meters of length of 1 pixel (float)
    :param outputShape: path of output shapefile (char)
    :returns: Dataset of new features assigned (ogr.Dataset)
    '''
    driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
    shadowDS = driver.Open(shadowShape)
    pixelWidth = pixelWidth
    pixelHeight = pixelWidth
    shadowLayer = shadowDS.GetLayer()
    shadowFeaturesCount = shadowLayer.GetFeatureCount()
    if outShape == '':
        driver = osgeo.ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(outShape)
    outDS.CopyLayer(shadowLayer,"Shadows")
    outLayer = outDS.GetLayer()
    shadow_def = osgeo.ogr.FieldDefn('Shadow_Len', osgeo.ogr.OFTInteger)
    height_def = osgeo.ogr.FieldDefn('Height',osgeo.ogr.OFTReal)
    outLayer.CreateField(shadow_def)
    outLayer.CreateField(height_def)
    for i in range(0,shadowFeaturesCount):
        #Convert Shape to Raster
        shadowFeature = shadowLayer.GetFeature(i)
        x_min, x_max, y_min, y_max = WindowsMaker(shadowFeature).make_coordinates()
        cols = int(math.ceil(float((x_max-x_min)) / float(pixelWidth))) #definition of the dimensions according to the extent and pixel resolution
        rows = int(math.ceil(float((y_max-y_min)) / float(pixelHeight)))
        rasterDS = gdal.GetDriverByName('MEM').Create("", cols,rows, 1, GDT_UInt16)
        rasterDS.SetGeoTransform((x_min, pixelWidth, 0,y_max, 0, -pixelHeight))
        rasterDS.SetProjection(shadowLayer.GetSpatialRef().ExportToWkt())
        gdal.RasterizeLayer(rasterDS,[1], shadowLayer,burn_values=[1])
        #Make Band List
        inband = rasterDS.GetRasterBand(1) 
        band_list = inband.ReadAsArray().astype(np.uint16)
        #Process Raster with Scipy
        w = ndimage.morphology.binary_fill_holes(band_list)
        band_list = ndimage.binary_opening(w, structure=np.ones((3,3))).astype(np.uint16)
        #Calculate zone
        prj = rasterDS.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        zone = int(str(srs.GetAttrValue('projcs')).split('Zone')[1][1:-1])
        #Calculate shadow Lenght
        geo_transform = rasterDS.GetGeoTransform()
        easting = geo_transform[0]
        northing = geo_transform[3]
        a = utm2wgs84(easting, northing, zone)
        lon,lat = a[0],a[1]
        #lat,lon = northing, easting
        shadowLen = shadow_length(band_list,lat,lon,'2012/8/11 9:35:00')
        buildingHeight = building_height(lat,lon,'2012/8/11 9:35:00',shadowLen)
        print "buildingHeight: {}\nshadowLen: {}".format(buildingHeight,shadowLen)
        #insert value to init shape
        outFeature = outLayer.GetFeature(i)
        outFeature.SetField('Shadow_Len',int(shadowLen))
        outFeature.SetField('Height',buildingHeight)
        outLayer.SetFeature(outFeature)
    return outDS

if __name__ == "__main__":
    if os.path.isfile("/home/gale/Izmir/final_building/tmp.shp"):
        os.remove("/home/gale/Izmir/final_building/tmp.shp")
    if os.path.isfile("/home/gale/Izmir/final_building/prova.shp"):
        os.remove("/home/gale/Izmir/final_building/prova.shp")
    height("/home/gale/Izmir/final_building/shadows.shp",0.5,0.5,outShape='/home/gale/Izmir/final_building/tmp.shp')
    #shadow_checker("/home/gale/Izmir/final_building/pan_class_6.shp","/home/gale/Izmir/final_building/tmp.shp",'2012/8/11 7:35:00', outputShape="/home/gale/Izmir/final_building/prova.shp", idfield="ID", resize=1)
