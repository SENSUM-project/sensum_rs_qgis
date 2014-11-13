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
from scipy import stats

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
    spatial_filter = bool(arg.spatial_filter)
    change_detection(main_folder,extraction,field,spatial_filter)


def args():
    parser = argparse.ArgumentParser(description='Change Detection')
    parser.add_argument("main_folder", help="????")
    parser.add_argument("extraction", help="????")
    parser.add_argument("field", help="????")
    parser.add_argument("--spatial_filter", default=False, const=True, nargs='?', help="????")
    args = parser.parse_args()
    return args


def change_detection(sat_folder,extraction,field="UrbanClass",spatial_filter=False):
    factors = {"changes":1, "position_change":2, "percentual_trues":0.5}
    class_str = "Mean1"
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
            target_layer = data_set.GetLayer()
            #print target_dir
            urban_class = get_class_from_shape(target_layer,field)
            urban_classes.append(urban_class)
            target_list = [(True if int(target_layer.GetFeature(i).GetField(class_str)) in urban_classes[target_index] else False) for i in range(target_layer.GetFeatureCount())]
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
        ref_list = [(True if int(ref_layer.GetFeature(i).GetField(class_str)) in urban_classes[0] else False) for i in range(ref_layer.GetFeatureCount())]
        urban_segments = np.vstack((urban_segments,ref_list))
        urban_segments = np.flipud(urban_segments)
        features_shape = urban_segments.shape[1]
        years_shape = urban_segments.shape[0]
        position_change_list = np.zeros(features_shape) #Lista che contiene l'anno in cui e' avvenuto l'ultimo cambiamento
        result_list = np.zeros(features_shape) #Lista col fattore di stabilita' (0 no cambiamento, 0.5 livello di confidenza, 0.9 urbano, 1 sicuro urbano, np.nan errore iniziale)
        #Calcolo il numero di cambiamenti (changes_list) e della posizione dell'ultimo cambiamento (position_change_list)
        urban_segments_vertical = list(map(list,zip(*urban_segments)))
        #FILTER 1
        status = Bar(len(urban_segments_vertical),"FILTER 1")
        for segment_index,urban_segment in enumerate(urban_segments_vertical):
            status(segment_index+1)
            changes = Changes(urban_segment)
            if changes.n_changes == 1 and changes.changes[0] == "GOOD":
                result_list[segment_index] = 1
                position_change_list[segment_index] = changes.position[0]
            elif changes.n_changes >= 1:
                result_list[segment_index] = 0.5
        #FILTER 2
        status = Bar(len(urban_segments_vertical),"FILTER 2")
        for segment_index in range(len(urban_segments_vertical)):
            status(segment_index+1)
            segment_value = urban_segments_vertical[segment_index]
            changes = Changes(segment_value).n_changes
            if changes <= round(len(segment_value)/2) and result_list[segment_index] == 0.5:
                for c in range(len(segment_value)):
                    if c+4 <= len(segment_value):
                        windows_four = segment_value[c:c+4]
                        changes = Changes(windows_four)
                        if changes.n_changes == 1:
                            if changes.changes[0] == "BAD":
                                if changes.mode:
                                    urban_segments_vertical[segment_index][c:c+4] = changes.mode
                                else:
                                    break
                            else:
                                continue
                        if changes.n_changes == 2:
                            if changes.position == [0,2]:
                                if windows_four == [0,1,1,0]:
                                    urban_segments_vertical[segment_index][c:c+4] = [1,1,1,0]
                                else:
                                    urban_segments_vertical[segment_index][c:c+4] = [1,0,0,0]
                            else:
                                urban_segments_vertical[segment_index][c:c+4] = changes.mode
                        if changes.n_changes == 3:
                            break
        status = Bar(len(urban_segments_vertical),"FILTER 2")
        for segment_index,urban_segment in enumerate(urban_segments_vertical):
            status(segment_index+1)
            changes = Changes(urban_segment)
            if result_list[segment_index] == 0.5:
                if changes.n_changes == 0:
                    result_list[segment_index] = 0.1
                    position_change_list[segment_index] = len(urban_segment)-1
                elif changes.n_changes == 1 and changes.changes[0] == "GOOD":
                    result_list[segment_index] = 0.9
                    position_change_list[segment_index] = changes.position[0]
            if (result_list[segment_index] == 0 and target_list[segment_index]) or (result_list[segment_index] == 0.1 and target_list[segment_index]):
                position_change_list[segment_index] = years_shape-1
        #FILTER 3
        if spatial_filter == True:
            status = Bar(len(urban_segments_vertical),"FILTER 3")
            for segment_index in range(features_shape):
                status(segment_index+1)
                if result_list[segment_index] == 0.5:
                    maker = CheckAround(ref_layer.GetFeature(segment_index),10)
                    maker.make_feature()
                    IDs = maker.get_ids(ref_layer,segment_index)
                    IDs_filtered = [i for i in IDs if result_list[i] != 0.5]
                    factor = float(len(IDs_filtered)/len(IDs))
                    fix_list = list()
                    dates = list()
                    current_list = list()
                    if factor >= 0.5:
                        for i in range(years_shape):
                            current_list.append(urban_segments[i][segment_index])
                        for si in IDs_filtered:
                            max_list = list()
                            for i in range(years_shape):
                                max_list.append(urban_segments[i][si])
                            dates.append(position_change_list[si])
                            fix_list.append(max_list) 
                    if fix_list:
                        if np.count_nonzero(np.logical_xor(current_list,stats.mode(fix_list)[0][0])) <= round(years_shape)/2:
                            position_change_list[segment_index] = stats.mode(dates)[0][0]
                            result_list[segment_index] = 0.8
        print "\n++++++++ TOTALE {} ++++++++".format(len(result_list))
        print "SICURO CAMBIAMENTO URBANO = {}".format(len([result for result in result_list if result == 1]))
        print "VALORE CAMBIATO STEP 1 NON CAMBIATO = {}".format(len([result for result in result_list if result == 0.1]))
        print "VALORE CAMBIATO STEP 1 CAMBIATO = {}".format(len([result for result in result_list if result == 0.9]))
        print "VALORE CAMBIATO STEP 2 = {}".format(len([result for result in result_list if result == 0.8]))
        print "LIVELLO DI CONFIDENZA = {}".format(len([result for result in result_list if result == 0.5]))
        print "NO CAMBIAMENTO = {}".format(len([result for result in result_list if result == 0]))
        print "ERRORE INIZIALE = {}".format(len([result for result in result_list if np.isnan(result)]))
        #Scrivo shapefile
        if extraction == "Dissimilarity":
            output_shape = sat_folder+"change_detection_dissimilarity.shp"
        elif extraction == "PCA":
            output_shape = sat_folder+"change_detection_pca.shp"
        if os.path.isfile(output_shape): os.remove(output_shape)
        output_data_set = driver.CreateDataSource(output_shape)
        output_data_set.CopyLayer(ref_layer,"")
        output_layer = output_data_set.GetLayer()
        output_layer.CreateField(ogr.FieldDefn("C_Factor", ogr.OFTReal))
        output_layer.CreateField(ogr.FieldDefn("C_Date", ogr.OFTInteger))
        output_layer.CreateField(ogr.FieldDefn("C_SEQUENCE", ogr.OFTString))
        for features_index in range(output_layer.GetFeatureCount()):
            feature = output_layer.GetFeature(features_index)
            stability_factor = result_list[features_index]
            feature.SetField("C_Factor", stability_factor)
            feature.SetField("C_SEQUENCE", "".join(map(str,urban_segments_vertical[features_index])))
            if (stability_factor == 0 and target_list[features_index]) or (stability_factor == 0.1 and target_list[features_index]):
                feature.SetField("C_Date", years[-1])
            if stability_factor > 0.5:
                feature.SetField("C_Date", years[int(position_change_list[features_index])])
            output_layer.SetFeature(feature)

def get_class_from_shape(target_layer,field="UrbanClass"):
    out_list = str(target_layer.GetFeature(0).GetField(field)).split(",")
    #print out_list
    out_list = map(int,out_list)
    return out_list

def get_class(target_layer):
    urban_classes_tmp = []
    dissimilarity1_sums = []
    dissimilarity2_sums = []
    dissimilarity3_sums = []
    counters = []
    for i in range(target_layer.GetFeatureCount()):
        urban_class = target_layer.GetFeature(i).GetField("Class")
        dissimilarity1 = target_layer.GetFeature(i).GetField(class_str)
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

class CheckAround(WindowsMaker):

    def get_ids(self,inLayer,inID):
        n_features = inLayer.GetFeatureCount()
        ID_list = list()
        for i in range(n_features):
            inFeature = inLayer.GetFeature(i)
            if self.windowFeature.GetGeometryRef().Intersect(inFeature.GetGeometryRef()) and inID != i:
                ID_list.append(i)
        return ID_list

class Changes(object):

    def __init__(self,input_array):
        self.list = input_array
        self.changes = list()
        self.position = list()
        self.stability = list()
        stability = 0
        for i in range(len(self.list)):
            try:
                if self.list[i] != self.list[i+1]:
                    self.changes.append(("GOOD" if self.list[i] else "BAD"))
                    self.position.append(i)
                    self.stability.append(stability)
                    stability = 0
                else:
                    stability = stability + 1
            except:
                break
        self.n_changes = len(self.changes)
        self.mode = ([int(stats.mode(self.list)[0][0])]*len(self.list) if stats.mode(self.list)[1][0] != len(self.list)/2 else False)

if __name__ == "__main__":
    main()
