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
from numpy.fft import fft2, ifft2, fftshift
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
    arg = args()
    main_folder = str(arg.main_folder)
    field = str(arg.field)
    extraction = str(arg.extraction)
    change_detection(main_folder,extraction,field)


def args():
    parser = argparse.ArgumentParser(description='Change Detection')
    parser.add_argument("main_folder", help="????")
    parser.add_argument("extraction", help="????")
    parser.add_argument("field", help="????")
    args = parser.parse_args()
    return args


def change_detection(sat_folder,extraction,field="UrbanClass"):
    factors = {"changes":1, "position_change":2, "percentual_trues":0.5}
    urban_classes = []
    years = []
    ####STACK SATELLITE CODE####
    #sat_folder = "/home/gale/Van_process"
    change_detection_method_EUCENTRE = True
    ref_dir = None
    if os.name == 'posix':
        separator = '/'
    else:
        separator = '\\'
    sat_folder = sat_folder + separator     
    dirs = os.listdir(sat_folder)
    dirs.sort()
    dirs = [dir for dir in dirs if os.path.isdir(sat_folder+dir)]
    if ref_dir is None or ref_dir == '': ref_dir = sat_folder+dirs[-1]+separator
    target_directories = (sat_folder+directory+separator for directory in reversed(dirs) if not os.path.isfile(sat_folder+directory) and (ref_dir!=sat_folder+directory+separator))
    for target_index,target_dir in enumerate(target_directories):
        years.append(os.path.basename(os.path.normpath(target_dir))[0:4])
    ####END STACK SATELLITE CODE####
        if change_detection_method_EUCENTRE:
            #Creo la matrice urban_segments con associato True o False a seconda che il segmento sia urbano o no per ogni anno
            data_set = (osgeo.ogr.GetDriverByName("ESRI Shapefile").Open(target_dir+'dissimilarity_class.shp') if extraction == "Dissimilarity" else osgeo.ogr.GetDriverByName("ESRI Shapefile").Open(target_dir+'pca_class.shp'))
            #data_set =  osgeo.ogr.GetDriverByName("ESRI Shapefile").Open(target_dir+'dissimilarity_class.shp')
            target_layer = data_set.GetLayer()
            #urban_class = get_class(target_layer)
            urban_class = get_class_from_shape(target_layer,field)
            urban_classes.append(urban_class)
            target_list = [(True if int(target_layer.GetFeature(i).GetField("Mean1")) == urban_classes[target_index] else False) for i in range(target_layer.GetFeatureCount())]
            urban_segments = (np.vstack((target_list,urban_segments)) if target_index else target_list)
    if change_detection_method_EUCENTRE:
        years.insert(0,os.path.basename(os.path.normpath(ref_dir))[0:4])
        driver = osgeo.ogr.GetDriverByName("ESRI Shapefile")
        ref_data_set = (osgeo.ogr.GetDriverByName("ESRI Shapefile").Open(ref_dir+'dissimilarity_class.shp') if extraction == "Dissimilarity" else osgeo.ogr.GetDriverByName("ESRI Shapefile").Open(ref_dir+'pca_class.shp'))
        #ref_data_set = driver.Open(ref_dir+'dissimilarity_class.shp')
        ref_layer = ref_data_set.GetLayer()
        #urban_class = get_class(ref_layer)
        urban_class = get_class_from_shape(ref_layer,field)
        urban_classes.insert(0, urban_class)
        #print urban_classes
        ref_list = [(True if int(ref_layer.GetFeature(i).GetField("Mean1")) == urban_classes[0] else False) for i in range(ref_layer.GetFeatureCount())]
        urban_segments = np.vstack((urban_segments,ref_list))
        features_shape = urban_segments.shape[1]
        years_shape = urban_segments.shape[0]
        changes_list = np.zeros(features_shape) #Lista che contiene il numero di cambiamenti per ogni segmento
        position_change_list = np.zeros(features_shape) #Lista che contiene l'anno in cui e' avvenuto l'ultimo cambiamento
        result_list = np.zeros(features_shape) #Lista col fattore di stabilita' (0 no cambiamento, 0.5 livello di confidenza, 0.9 urbano, 1 sicuro urbano, np.nan errore iniziale)
        #Calcolo il numero di cambiamenti (changes_list) e della posizione dell'ultimo cambiamento (position_change_list)
        for year_index,segments in enumerate(urban_segments):
            for segment_index,segment in enumerate(segments):
                if year_index:
                    if urban_segments[year_index-1][segment_index] != segment:
                        if not changes_list[segment_index]:
                            position_change_list[segment_index] = year_index
                        changes_list[segment_index] = changes_list[segment_index] + 1
        #Calcolo il risultato
        urban_segments_vertical = list(zip(*urban_segments))
        for segment_index,changes in enumerate(changes_list):
            if changes > factors["changes"]:
                percentual_trues = float(urban_segments_vertical[segment_index].count(True))/years_shape
                if percentual_trues > factors["percentual_trues"] and urban_segments[0][segment_index] == False: result_list[segment_index] = 0.8
                elif percentual_trues < 1-factors["percentual_trues"] and urban_segments[0][segment_index] == True: pass
                else: result_list[segment_index] = 0.5 #LIVELLO DI CONFIDENZA
            elif changes <= factors["changes"] and changes > 0:
                if urban_segments[0][segment_index] == True and urban_segments[-1][segment_index] == False and position_change_list[segment_index] < factors["position_change"]:
                    result_list[segment_index] = None #ERRORE INIZIALE
                elif position_change_list[segment_index] == 1:
                    result_list[segment_index] = 1 #SICURO URBANO, EVOLUZIONE NEL TEMPO
                elif position_change_list[segment_index] > 1:
                    result_list[segment_index] = 0.9 #URBANO, EVOLUZIONE NEL TEMPO
        print "++++++++ TOTALE {} ++++++++".format(len(result_list))
        print "SICURO CAMBIAMENTO URBANO = {}".format(len([result for result in result_list if result == 1]))
        print "CAMBIAMENTO URBANO (0.9) = {}".format(len([result for result in result_list if result == 0.9]))
        print "CAMBIAMENTO URBANO (0.8) = {}".format(len([result for result in result_list if result == 0.8]))
        print "LIVELLO DI CONFIDENZA = {}".format(len([result for result in result_list if result == 0.5]))
        print "NO CAMBIAMENTO = {}".format(len([result for result in result_list if result == 0]))
        print "ERRORE INIZIALE = {}".format(len([result for result in result_list if np.isnan(result)]))
        #Scrivo shapefile
        if os.path.isfile(sat_folder+"change_detection.shp"): os.remove(sat_folder+"change_detection.shp")
        output_data_set = driver.CreateDataSource(sat_folder+"change_detection.shp")
        output_data_set.CopyLayer(ref_layer,"")
        output_layer = output_data_set.GetLayer()
        output_layer.CreateField(ogr.FieldDefn("C_Factor", ogr.OFTReal))
        output_layer.CreateField(ogr.FieldDefn("C_Date", ogr.OFTInteger))
        for features_index in range(output_layer.GetFeatureCount()):
            feature = output_layer.GetFeature(features_index)
            stability_factor = result_list[features_index]
            feature.SetField("C_Factor", stability_factor)
            if stability_factor == 0 and target_list[features_index]:
                feature.SetField("C_Date", years[-1])
            elif stability_factor > 0.5:
                feature.SetField("C_Date", years[int(position_change_list[features_index+1])])
            #if np.isnan(stability_factor):
                #feature.SetField("Class", urban_classes[0])
            output_layer.SetFeature(feature)
        print urban_classes, years

def get_class_from_shape(target_layer,field="UrbanClass"):
    return int(target_layer.GetFeature(0).GetField(field))

def get_class(target_layer):
    urban_classes_tmp = []
    dissimilarity1_sums = []
    dissimilarity2_sums = []
    dissimilarity3_sums = []
    counters = []
    for i in range(target_layer.GetFeatureCount()):
        urban_class = target_layer.GetFeature(i).GetField("Class")
        dissimilarity1 = target_layer.GetFeature(i).GetField("Mean1")
        dissimilarity2 = target_layer.GetFeature(i).GetField("Mean2")
        dissimilarity3 = target_layer.GetFeature(i).GetField("Mean3")
        if urban_class not in urban_classes_tmp:
            urban_classes_tmp.append(urban_class)
            counters.append(1)
            dissimilarity1_sums.append(dissimilarity1)
            dissimilarity2_sums.append(dissimilarity2)
            dissimilarity3_sums.append(dissimilarity3)
        else:
            index = (i for i,urban_class_tmp in enumerate(urban_classes_tmp) if urban_class_tmp == urban_class).next()
            counters[index] = counters[index] + 1
            dissimilarity1_sums[index] = dissimilarity1_sums[index] + dissimilarity1
            dissimilarity2_sums[index] = dissimilarity2_sums[index] + dissimilarity2
            dissimilarity3_sums[index] = dissimilarity3_sums[index] + dissimilarity3
    for i,urban_class in enumerate(urban_classes_tmp):
        dissimilarity1_sums[i] = dissimilarity1_sums[i]/counters[i]
        dissimilarity2_sums[i] = dissimilarity2_sums[i]/counters[i]
        dissimilarity3_sums[i] = dissimilarity3_sums[i]/counters[i]
    index1 = (urban_classes_tmp[i] for i,disimmilarity1_sum in enumerate(dissimilarity1_sums) if disimmilarity1_sum == max(dissimilarity1_sums)).next()
    index2 = (urban_classes_tmp[i] for i,disimmilarity2_sum in enumerate(dissimilarity2_sums) if disimmilarity2_sum == max(dissimilarity2_sums)).next()
    index3 = (urban_classes_tmp[i] for i,disimmilarity3_sum in enumerate(dissimilarity3_sums) if disimmilarity3_sum == max(dissimilarity3_sums)).next()
    index_list = [index1,index2,index3]
    return max(set(index_list), key=index_list.count)

if __name__ == "__main__":
    main()
