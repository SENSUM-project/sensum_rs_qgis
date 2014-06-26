'''
.. module:: classification
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
Last modified on Mar 09, 2014

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

import os,sys
import config
import numpy as np
import scipy.stats
import osgeo.gdal
import osgeo.ogr
import shutil
import cv2
import xml.etree.cElementTree as ET
import otbApplication
from conversion import *
import time
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import operator

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'


def unsupervised_classification_otb(input_raster,output_raster,n_classes,n_iterations):
    
    '''Unsupervised K-Means classification using OTB library.
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :param n_classes: number of classes to extract (integer)
    :param n_iterations: number of iterations of the classifier (integer)
    :returns:  an output raster is created
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 20/03/2014
    '''
    
    KMeansClassification = otbApplication.Registry.CreateApplication("KMeansClassification") 
 
    # The following lines set all the application parameters: 
    KMeansClassification.SetParameterString("in", input_raster) 
    KMeansClassification.SetParameterInt("ts", 1000) 
    KMeansClassification.SetParameterInt("nc", n_classes) 
    KMeansClassification.SetParameterInt("maxit", n_iterations) 
    KMeansClassification.SetParameterFloat("ct", 0.0001) 
    KMeansClassification.SetParameterString("out", output_raster) 
    
    # The following line execute the application 
    KMeansClassification.ExecuteAndWriteOutput()
    
    
def unsupervised_classification_opencv(input_band_list,n_classes,n_iterations):
    
    '''Unsupervised K-Means classification using OpenCV library.
    
    :param input_band_list: list of 2darrays corresponding to bands (band 1: blue) (list of numpy arrays)
    :param n_classes: number of classes to extract (integer)
    :param n_iterations: number of iterations of the classifier (integer)
    :returns:  an output 2darray is created with the results of the classifier
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 20/03/2014
    '''
    
    img = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2],input_band_list[3])) #stack the 4 bands together
    Z = img.reshape((-1,4)) #reshape for the classifier
    
    Z = np.float32(Z) #convert to np.float32
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iterations, 0.0001) #definition of the criteria
    ret,label,center=cv2.kmeans(Z,n_classes,criteria,n_iterations,cv2.KMEANS_RANDOM_CENTERS) #kmeans classification
    center = np.uint8(center) 
    res = center[label.flatten()]
    res2 = res[:,0] #extraction of the desired row
    output_array = res2.reshape(input_band_list[0].shape) #reshape to original raster dimensions
    
    return output_array
    
    
def train_classifier_otb(input_raster_list,input_shape_list,output_txt,classification_type,training_field):
    
    '''Training of the desired classifier using OTB library
    
    :param input_raster_list: list of paths and names of the input raster files (*.TIF,*.tiff) (list of strings)
    :param input_shape_list: list of paths and names of the input shapefiles containing the training sets (*.TIF,*.tiff) (list of strings)
    :param output_txt: path and name of text file with the training parameters (*.txt) (string)
    :param classification type: definition of the desired classification algorithm ('libsvm','svm','dt','gbt','bayes','rf','knn') (string)
    :param training_field: name of the discriminan attribute in the training shapefile (string)
    :returns:  an output text file is created along with a csv file containing a confusion matrix
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu118.html#x152-8600005.8.8
    '''
    root = ET.Element("FeatureStatistics")
    
    #XML file creation as input for OTB
    print 'Number of provided raster files: ' + str(len(input_raster_list))
    
    for i in range(0,len(input_raster_list)):
        rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster_list[i])
        band_list = read_image(input_raster_list[i],np.uint16,0)
        statistic = ET.SubElement(root,"Statistic")
        statistic.set("name","mean")
        for b in range(0,nbands):
            statistic_vector = ET.SubElement(statistic,"StatisticVector")
            statistic_vector.set("value",str(round(np.mean(band_list[b]),4)))
      
          
    for i in range(0,len(input_raster_list)):
        band_list = read_image(input_raster_list[i],np.uint16,0)
        statistic = ET.SubElement(root,"Statistic")
        statistic.set("name","stddev")
        for b in range(0,nbands):
            statistic_vector = ET.SubElement(statistic,"StatisticVector")
            statistic_vector.set("value",str(round(np.std(band_list[b])/2,4)))
        
    tree = ET.ElementTree(root)
    tree.write(input_raster_list[0][:-4]+'_statistics.xml')
    
    #OTB Train Classifier
    TrainImagesClassifier = otbApplication.Registry.CreateApplication("TrainImagesClassifier") 
     
    # The following lines set all the application parameters: 
    TrainImagesClassifier.SetParameterStringList("io.il", input_raster_list) 
    TrainImagesClassifier.SetParameterStringList("io.vd", input_shape_list) 
    TrainImagesClassifier.SetParameterString("io.imstat", input_raster_list[0][:-4]+'_statistics.xml') 
    TrainImagesClassifier.SetParameterInt("sample.mv", 100) 
    TrainImagesClassifier.SetParameterInt("sample.mt", 100) 
    TrainImagesClassifier.SetParameterFloat("sample.vtr", 0.5) 
    TrainImagesClassifier.SetParameterString("sample.edg","1") 
    TrainImagesClassifier.SetParameterString("sample.vfn", training_field)
    TrainImagesClassifier.SetParameterString("classifier",classification_type) 
    TrainImagesClassifier.SetParameterString("io.out", output_txt)  
    TrainImagesClassifier.SetParameterString("io.confmatout", output_txt[:-4] + "_ConfusionMatrix.csv") 
    
    # The following line execute the application 
    TrainImagesClassifier.ExecuteAndWriteOutput()
 

def supervised_classification_otb(input_raster,input_txt,output_raster):
    
    '''Supervised classification using OTB library
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param input_txt: path and name of text file with the training parameters (*.txt) (string)
    :param output_raster: path and name of the output raster file (*.TIF,*.tiff) (string)
    :returns:  an output raster file is created with the results of the classification
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 22/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu115.html#x149-8410005.8.5
    '''
    
    #XML file creation as input for OTB. File has to be re-generated in case of training file produced with different input file
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster)
    band_list = read_image(input_raster,np.uint16,0)
    
    root = ET.Element("FeatureStatistics")
    statistic = ET.SubElement(root,"Statistic")
    statistic.set("name","mean")
    
    for b in range(0,nbands):
        statistic_vector = ET.SubElement(statistic,"StatisticVector")
        statistic_vector.set("value",str(round(np.mean(band_list[b]),4)))
    
    statistic = ET.SubElement(root,"Statistic")
    statistic.set("name","stddev")
    for b in range(0,nbands):
        statistic_vector = ET.SubElement(statistic,"StatisticVector")
        statistic_vector.set("value",str(round(np.std(band_list[b])/2,4)))
    
    tree = ET.ElementTree(root)
    tree.write(input_raster[:-4]+'_statistics.xml')
    
    # The following line creates an instance of the ImageClassifier application 
    ImageClassifier = otbApplication.Registry.CreateApplication("ImageClassifier") 
    # The following lines set all the application parameters: 
    ImageClassifier.SetParameterString("in", input_raster) 
    ImageClassifier.SetParameterString("imstat", input_raster[:-4]+'_statistics.xml') 
    ImageClassifier.SetParameterString("model", input_txt) 
    ImageClassifier.SetParameterString("out", output_raster) 
    # The following line execute the application 
    ImageClassifier.ExecuteAndWriteOutput()
    

def generate_training(input_band_list,input_shape,training_field,pixel_width,pixel_height):
    
    '''Extract the training set from the input shapefile
    
    :param input_band_list: list of 2darrays corresponding to bands (band 1: blue) (list of numpy arrays)
    :param input_shape: path and name of shapefile with the polygons for training definition (*.shp) (string)
    :param training_field: name of the discriminant attribute in the training shapefile (string)
    :returns: list with 2 2darrays is returned (sample_matrix, train_matrix)
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 01/04/2014
    '''
    
    driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    inDS = driver_shape.Open(input_shape, 0)
    if inDS is None:
        print 'Could not open file'
        sys.exit(1)
    inLayer = inDS.GetLayer()
    numFeatures = inLayer.GetFeatureCount()
    print 'Number of reference features: ' + str(numFeatures)
    #temp_shape = input_shape[:-4]+'_temp.shp'
    
    sample_matrix = np.zeros((1,4)).astype(np.float32)
    train_matrix = np.zeros((1)).astype(np.int32)
    print sample_matrix.shape
    
    stack_new = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2],input_band_list[3])) #stack with the original bands
    
    for n in range(0,numFeatures):
        print 'Feature ' + str(n+1) + ' of ' + str(numFeatures)
        #separate each polygon creating a temp file
        temp = split_shape(inLayer,n) #extract single polygon
        inFeature = inLayer.GetFeature(n)
        training_def = inFeature.GetField(training_field) #take the class definition for the sample
        #conversion of the temp file to raster
        
        temp_layer = temp.GetLayer()
        reference_matrix, ref_geo_transform = polygon2array(temp_layer,pixel_width,pixel_height) #convert the single polygon to raster
        temp.Destroy() 
        
        mask = np.where(reference_matrix == 1) #raster used to extract the mask for the desired sample
    
        print 'Pixels per sample: ' + str(len(mask[0]))
        for l in range(0,len(mask[0])):
            sample_matrix = np.append(sample_matrix, [stack_new[mask[0][l]][mask[1][l]][:]], 0) #define the sample matrix with the rows = number of pixels from each sample and columns = number of bands
            train_matrix = np.append(train_matrix, [training_def], 0) #training set defined as an array with number of elements equal to rows of the sample matrix
    
    return sample_matrix,train_matrix


def create_training_file(sample_matrix,train_matrix,output_file):
    
    '''Export training set to text file
    
    :param sample_matrix: 2darray with number of rows equal to number of sample pixels and columns equal to number of bands
    :param train_matrix: 1darray with class value corresponding to each pixel sample
    :param output_file: path and name of the output text file (*.txt) (string)
    :returns:  an output file is created with structure sample,sample,sample,sample,class
    :raises: AttributeError, KeyError
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 01/04/2014
    '''
    
    train_file = train_matrix.reshape((train_matrix.size,1))
    train_file = np.hstack((sample_matrix, train_file))
    np.savetxt(output_file, train_file, delimiter=',')
    
    
def read_training_file(input_file):
    
    '''Read training set from text file
    
    :param input_file: path and name of the text file with the training set
    :returns:  list with 2 2darrays is returned (sample_matrix, train_matrix)
    :raises: AttributeError, KeyError
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 01/04/2014
    '''
    
    train_file = np.genfromtxt(input_file, delimiter=',')
    samples_from_file = np.float32(train_file[0:train_file.size,0:4]) #read samples from file
    train_from_file = np.float32(train_file[0:train_file.size,4]) #read defined classes from file
    
    return samples_from_file,train_from_file
    
    
def update_training_file(input_file,sample_matrix,train_matrix):
    
    '''Update a text file with new training samples
    
    :param input_file: path and name of the text file with the training set
    :param sample_matrix: 2darray with the samples to add 
    :param train_matrix: 1darray with the corresponding classes
    :returns:  input file is updated with new data
    :raises: AttributeError, KeyError
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 01/04/2014
    '''
    
    train_file = train_matrix.reshape((train_matrix.size,1))
    train_file = np.hstack((sample_matrix, train_file))
    train_file = np.vstack((np.genfromtxt(input_file, delimiter=','), train_file))
    np.savetxt(input_file, train_file, delimiter=',')
    
    
def supervised_classification_opencv(input_band_list,sample_matrix,train_matrix,classification_type):
    
    '''Supervised classification using OpenCV library
    
    :param input_band_list: list of 2darrays corresponding to bands (band 1: blue) (list of numpy arrays)
    :param sample_matrix: 2darray with the samples to add 
    :param train_matrix: 1darray with the corresponding classes
    :param classification type: definition of the desired classification algorithm ('svm','dt','gbt','bayes','rf','knn') (string)
    :returns:  2darray with predicted classes
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Daniel Aurelio Galeazzo - Mostapha Harb
    Last modified: 01/04/2014
    '''
    
    stack_new = np.dstack((input_band_list[0],input_band_list[1],input_band_list[2],input_band_list[3])) #stack with the original bands
    samples = stack_new.reshape((-1,4)) #input image as matrix with rows = (rows_original * cols_original) and columns = number of bands
    print classification_type
    
    if classification_type == "svm":
    
        params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC,C = 10000 ) #definition of the SVM parameters (kernel, type of algorithm and parameter related to the chosen algorithm
        cl = cv2.SVM()
        cl.train_auto(sample_matrix,train_matrix,None,None,params,100) #creation of the training set forcing the parameters optimization
        y_val = cl.predict_all(samples) #classification of the input image
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16) #reshape to the original rows and columns
    
    if classification_type == "dt":
        
        cl = tree.DecisionTreeClassifier()
        cl = cl.fit(sample_matrix, train_matrix)
        y_val = cl.predict(samples)
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16)
    
    if classification_type == "gbt":
        
        cl = GradientBoostingClassifier()
        cl = cl.fit(sample_matrix, train_matrix)
        y_val = cl.predict(samples)
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16)
    
    if classification_type == "bayes":
    
        cl = cv2.NormalBayesClassifier(sample_matrix, train_matrix)
        y_val = cl.predict(samples)
        y_val = np.array(y_val[1])
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16)
        
    if classification_type == "rf":
        
        cl = cv2.RTrees()
        cl.train(sample_matrix, cv2.CV_ROW_SAMPLE, train_matrix)
        y_val = np.float32( [cl.predict(s) for s in samples] )
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16)
        
    if classification_type == "knn":
        
        cl = cv2.KNearest()
        cl.train(sample_matrix, train_matrix)
        retval, results, neigh_resp, dists = cl.find_nearest(samples, k = 10)
        y_val = results.ravel()
        output = y_val.reshape(input_band_list[0].shape).astype(np.uint16)
    
    return output


def class_to_segments(input_raster,input_shape,output_shape):
    
    '''Assign the most frequent value inside a segment to the segment itself
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param input_shape: path and name of shapefile with the segmentation results (*.shp) (string)
    :param output_shape: path and name of the output shapefile (*.shp) (string)
    :returns:  an output shapefile is created with a new attribute field related to the most frequent value inside the segment
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 23/03/2014
    '''
    #Example of hybrid approach
    #TODO: this is only a spatial union operation, isn't it? So it is not part of the hybrid approach where you aggregate pixel classes to segments!?
    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster) 
    band_list_class = read_image(input_raster,np.int32,0) #read original raster file
    shp2rast(input_shape,input_shape[:-4]+'.TIF',rows,cols,'DN',0,0,0,0,0,0) #conversion of the segmentation results from shape to raster for further processing
    band_list_seg = read_image(input_shape[:-4]+'.TIF',np.int32,0) #read segmentation raster file
    
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    infile=driver_shape.Open(input_shape)
    inlayer=infile.GetLayer()
    outfile=driver_shape.CreateDataSource(output_shape)
    outlayer=outfile.CreateLayer('Features',geom_type=osgeo.ogr.wkbPolygon)
    
    layer_defn = inlayer.GetLayerDefn()
    infeature = inlayer.GetNextFeature()
    feature_def = outlayer.GetLayerDefn()
    dn_def = osgeo.ogr.FieldDefn('DN', osgeo.ogr.OFTInteger)
    class_def = osgeo.ogr.FieldDefn('Class',osgeo.ogr.OFTInteger)
    area_def = osgeo.ogr.FieldDefn('Area',osgeo.ogr.OFTReal)
    
    outlayer.CreateField(dn_def)
    outlayer.CreateField(class_def)
    outlayer.CreateField(area_def)
    
    n_feature = inlayer.GetFeatureCount()
    j = 1
    while infeature:
        print str(j) + ' of ' + str(n_feature)
        j = j+1
        dn = infeature.GetField('DN')
        # get the input geometry
        geom = infeature.GetGeometryRef()
        area = geom.Area()
        # create a new feature
        outfeature = osgeo.ogr.Feature(feature_def)
        # set the geometry and attribute
        outfeature.SetGeometry(geom)
        seg_pos = np.where(band_list_seg[0] == dn) #returns a list of x and y coordinates related to the pixels satisfying the given condition
        mat_pos = np.zeros(len(seg_pos[0]))
        
        #Extract all the pixels inside a segment
        for l in range(0,len(seg_pos[0])):
            mat_pos[l] = band_list_class[0][seg_pos[0][l]][seg_pos[1][l]]
        
        mode_ar = scipy.stats.mode(mat_pos)
        mode = mode_ar[0][0]
        
        outfeature.SetField('DN',dn)
        outfeature.SetField('Class',mode)
        outfeature.SetField('Area',area)
        outlayer.CreateFeature(outfeature)
        outfeature.Destroy() 
        infeature = inlayer.GetNextFeature()
    
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()    
    
    shutil.copyfile(input_shape[:-4]+'.prj', output_shape[:-4]+'.prj') #projection definition
    

def confusion_matrix(input_raster,input_shape,reference_field,output_file):    
    
    '''Compute a confusion matrix for accuracy estimation of the classification
    
    :param input_raster: path and name of the input raster file (*.TIF,*.tiff) (string)
    :param input_shape: path and name of shapefile with the reference polygons (*.shp) (string)
    :param reference_field: name of the discriminant attribute in the reference shapefile (string)
    :param output_file: path and name of the output csv file (*.csv) (string)
    :returns:  an output csv file is created containing the confusion matrix with rows as reference labels and columns as produced labels
    :raises: AttributeError, KeyError
    
    Author: Daniele De Vecchi - Mostapha Harb
    Last modified: 23/03/2014
    
    Reference: http://orfeo-toolbox.org/CookBook/CookBooksu112.html#x146-8070005.8.2
    '''
    
    ComputeConfusionMatrix = otbApplication.Registry.CreateApplication("ComputeConfusionMatrix") 
    ComputeConfusionMatrix.SetParameterString("in", input_raster) 
    ComputeConfusionMatrix.SetParameterString("out", output_file) 
    ComputeConfusionMatrix.SetParameterString("ref","vector") 
    ComputeConfusionMatrix.SetParameterString("ref.vector.in", input_shape) 
    ComputeConfusionMatrix.SetParameterString("ref.vector.field", reference_field) 
    ComputeConfusionMatrix.SetParameterInt("nodatalabel", 255) 
     
    # The following line execute the application 
    ComputeConfusionMatrix.ExecuteAndWriteOutput()
    

def extract_from_shape(input_shape,output_shape='',field_name='Class',*classes):

    '''Extract a subset of the input shapefile according to the specified attribute field and list of values
    
    :param input_shape: path and name of the input shapefile (*.shp) (string)
    :param classes: value of class which want to extract (string)
    :param output_shape: path and name of the output shapefile (*.shp) (string)
    :param field_name: name of field to extract (string)
    :returns:  an output layer is created as a subset of the original shapefile
    :raises: AttributeError, KeyError
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''

    driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
    inDS = driver.Open(input_shape)

    path,file = os.path.split(input_shape)
    query = 'SELECT * FROM ' + str(file[:-4]) + ' WHERE '
    for q in range(len(classes)):
        if q == 0:
            query = query + '({} = '.format(field_name) + str(classes[q]) + ')' 
        else:
            query = query + ' OR ({} = '.format(field_name) + str(classes[q]) + ')' 
    inLayer = inDS.ExecuteSQL(query)

    if output_shape == '':
        driver = osgeo.ogr.GetDriverByName("Memory")
    outDS = driver.CreateDataSource(output_shape)
    outDS.CopyLayer(inLayer,"Shadows")
    return outDS


def reclassify_raster(input_band,*operations):
    
    '''Reclassify results of a classification according to the operation list
    
    :param input_band: 2darray corresponding to single classification band (numpy array)
    :param operations: list of operations to apply (e.g. '0 where 3', '255 where >3') (strings)
    :returns: output 2darray is created with the results of the reclassification process
    :raises: AttributeError, KeyError
    
    Author: Daniel Aurelio Galeazzo - Daniele De Vecchi - Mostapha Harb
    Last modified: 23/05/2014
    '''
    ops = ["<",">"]
    output_band = np.array(input_band)
    output_band_list = []
    logic_list = []
    for operation in operations:
        value, expression = operation.split(" where ")
        op = [op for op in ops if op in expression]
        if op:
            op = str(op[0])
            cls = int(expression.strip(op))
            if op == '<':
                output_band_list.append(np.where(output_band > cls, output_band, value))
                logic_list.append(np.where(output_band > cls, 0, 1))
            else:
                output_band_list.append(np.where(output_band < cls, output_band, value))
                logic_list.append(np.where(output_band < cls, 0, 1))
        else:
            cls = int(expression)
            output_band_list.append(np.where(output_band != cls, output_band, value))
            logic_list.append(np.where(output_band != cls, 0, 1))
    for i in range(len(operations)):
        output_band = np.where(logic_list[i] != 1, output_band,output_band_list[i])
    print output_band
