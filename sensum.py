# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sensum
                                 A QGIS plugin
 Sensum QGIS Plugin
                              -------------------
        begin                : 2014-05-27
        copyright            : (C) 2014 by Eucentre
        email                : dgaleazzo@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
# Import the PyQt and QGIS libraries
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from qgis.core import *
# Initialize Qt resources from file resources.py
import resources_rc
# Import the code for the dialog
from sensumdialog import *
import os,sys
import os.path
import shutil
import time
import tempfile
import osgeo.gdal, gdal
import osgeo.ogr
import numpy as np
import math
import scipy.signal
import scipy as sp
import cv2
import subprocess
from osgeo.gdalconst import *
from sensum_library.preprocess import *
from sensum_library.classification import *
from sensum_library.segmentation import *
from sensum_library.conversion import *
from sensum_library.segmentation_opt import *
from sensum_library.features import *
from sensum_library.secondary_indicators import *
from sensum_library.multi import *
from numpy.fft import fft2, ifft2, fftshift
from skimage.morphology import square, closing
import otbApplication
from scripts.utils import *

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

def string_qmark(string):
    try: return "\""+string+"\""
    except: return None

def executeScript(command, progress=None):
    if os.name != "posix":
        # Found at http://stackoverflow.com/questions/5069224/handling-subprocess-crash-in-windows
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen's response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002 # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
        subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
    else:
        subprocess_flags = 0

    command = (os.path.dirname(os.path.abspath(__file__))+command if os.name == "posix" else 'C:/Python27/python.exe "'+os.path.dirname(os.path.abspath(__file__))+command)
    #QMessageBox.information(None, "Info", command)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess_flags,
        universal_newlines=True,
        ).stdout
    for line in iter(proc.readline, ''):
        if '[*' in line:
            idx = line.find('[*')
            perc = line[idx:(idx+102)].count("*")
            status = line[line.find('STATUS: ')+8:idx]
            print status
            if perc != 0 and progress:
                progress.progressBar.setValue(perc)
                progress.label_title.setText(status)
        QtGui.qApp.processEvents()

def executeOtb(command, progress=None,label = "OTB library recalled"):
    if os.name != "posix":
        bit = ("64" if os.path.isdir("C:/OSGeo4W64") else "")
        osgeopath = "C:/OSGeo4W{}/bin/".format(bit)
        command = osgeopath + command
    QMessageBox.information(None, "Info", command)    
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        ).stdout
    for line in iter(proc.readline, ''):
        if '[*' in line:
            idx = line.find('[*')
            perc = int(line[idx - 4:idx - 2].strip(' '))
            if perc != 0 and progress:
                progress.progressBar.setValue(perc)
                progress.label_title.setText(label)
        QtGui.qApp.processEvents()

def segmentation_progress(input_raster,output_file,filter,paramaters=None,progress=None):
    paramaters_string = ""
    if paramaters:
        for paramater in paramaters.keys():
            value = paramaters.get(paramater)
            if value:
                paramaters_string = paramaters_string + "-filter.{}.{} {} ".format(filter,paramater,value)
    command = "otbcli_Segmentation -progress 1 {} -in {} -filter {} -mode.vector.out {}".format(paramaters_string,input_raster,filter,output_file)
    executeOtb(command, progress,filter)

def parse_input(string):
    return string.replace("[Choose from a file..] ","")

class Sensum:

    def __init__(self, iface):
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value("locale/userLocale")[0:2]
        localePath = os.path.join(self.plugin_dir, 'i18n', 'sensum_{}.qm'.format(locale))
        if os.path.exists(localePath):
            self.translator = QTranslator()
            self.translator.load(localePath)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

    def initGui(self):

        self.toolBar = self.iface.addToolBar("SENSUM")

        ######################
        ## PANSHARP
        ######################
        # Create action that will start plugin configuration
        self.action_pansharp = QAction(
            QIcon(":/plugins/sensum_plugin/icons/pansharp.png"),
            u"Pansharp", self.iface.mainWindow())
        # connect the action to the run method
        self.action_pansharp.triggered.connect(self.pansharp)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_pansharp)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_pansharp)
        
        ######################
        ## CLASSIFICATION
        ######################
        # Create action that will start plugin configuration
        self.action_classification = QAction(
            QIcon(":/plugins/sensum_plugin/icons/classification.png"),
            u"Classification", self.iface.mainWindow())
        # connect the action to the run method
        self.action_classification.triggered.connect(self.classification)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_classification)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_classification)

        ######################
        ## SEGMENTATION
        ######################
        # Create action that will start plugin configuration
        self.action_segmentation = QAction(
            QIcon(":/plugins/sensum_plugin/icons/segmentation.png"),
            u"Segmentation", self.iface.mainWindow())
        # connect the action to the run method
        self.action_segmentation.triggered.connect(self.segmentation)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_segmentation)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_segmentation)

        ######################
        ## FEATURES
        ######################
        # Create action that will start plugin configuration
        self.action_features = QAction(
            QIcon(":/plugins/sensum_plugin/icons/features.png"),
            u"Features", self.iface.mainWindow())
        # connect the action to the run method
        self.action_features.triggered.connect(self.features)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_features)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_features)

        ######################
        ## BUILDING HEIGHT
        ######################
        # Create action that will start plugin configuration
        self.action_building_height = QAction(
            QIcon(":/plugins/sensum_plugin/icons/buildingsheight.png"),
            u"Building Height", self.iface.mainWindow())
        # connect the action to the run method
        self.action_building_height.triggered.connect(self.build_height)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_building_height)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_building_height)

        ######################
        ## CO-REGISTRATION
        ######################
        # Create action that will start plugin configuration
        self.action_coregistration = QAction(
            QIcon(":/plugins/sensum_plugin/icons/coregistration.png"),
            u"Coregistration", self.iface.mainWindow())
        # connect the action to the run method
        self.action_coregistration.triggered.connect(self.coregistration)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_coregistration)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_coregistration)

        ######################
        ## FOOTPRINTS
        ######################
        # Create action that will start plugin configuration
        self.action_footprints = QAction(
            QIcon(":/plugins/sensum_plugin/icons/footprints.png"),
            u"Footprints", self.iface.mainWindow())
        # connect the action to the run method
        self.action_footprints.triggered.connect(self.footprints)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_footprints)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_footprints)

        ######################
        ## STACK SATELLITE
        ######################
        # Create action that will start plugin configuration
        self.action_stacksatellite = QAction(
            QIcon(":/plugins/sensum_plugin/icons/stacksatellite.png"),
            u"Stack Satellite", self.iface.mainWindow())
        # connect the action to the run method
        self.action_stacksatellite.triggered.connect(self.stacksatellite)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_stacksatellite)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_stacksatellite)

        ######################
        ## DENSITY
        ######################
        # Create action that will start plugin configuration
        self.action_density = QAction(
            QIcon(":/plugins/sensum_plugin/icons/density.png"),
            u"Density", self.iface.mainWindow())
        # connect the action to the run method
        self.action_density.triggered.connect(self.density)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_density)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_density)

        ######################
        ## INDEXES
        ######################
        # Create action that will start plugin configuration
        self.action_temporal = QAction(
            QIcon(":/plugins/sensum_plugin/icons/temporal.png"),
            u"Temporal Analysis", self.iface.mainWindow())
        # connect the action to the run method
        self.action_temporal.triggered.connect(self.temporal)
        # Add toolbar button and menu item
        self.toolBar.addAction(self.action_temporal)
        self.iface.addPluginToMenu(u"&SENSUM", self.action_temporal)

    def unload(self):
        self.iface.removePluginMenu(u"&SENSUM", self.action_pansharp)
        self.iface.removeToolBarIcon(self.action_pansharp)
        self.iface.removePluginMenu(u"&SENSUM", self.action_classification)
        self.iface.removeToolBarIcon(self.action_classification)
        self.iface.removePluginMenu(u"&SENSUM", self.action_segmentation)
        self.iface.removeToolBarIcon(self.action_segmentation)
        self.iface.removePluginMenu(u"&SENSUM", self.action_features)
        self.iface.removeToolBarIcon(self.action_features)
        self.iface.removePluginMenu(u"&SENSUM", self.action_building_height)
        self.iface.removeToolBarIcon(self.action_building_height)
        self.iface.removePluginMenu(u"&SENSUM", self.action_coregistration)
        self.iface.removeToolBarIcon(self.action_coregistration)
        self.iface.removePluginMenu(u"&SENSUM", self.action_footprints)
        self.iface.removeToolBarIcon(self.action_footprints)
        self.iface.removePluginMenu(u"&SENSUM", self.action_stacksatellite)
        self.iface.removeToolBarIcon(self.action_stacksatellite)
        self.iface.removePluginMenu(u"&SENSUM", self.action_density)
        self.iface.removeToolBarIcon(self.action_density)

    def changeActive(self,comboBox):
        comboBox.clear()
        comboBox.addItem(_fromUtf8(""))
        comboBox.setItemText(0, _translate("Pansharp", "[Choose from a file..]", None))
        current_layer = self.iface.mapCanvas().currentLayer()
        layers = self.iface.mapCanvas().layers()
        for i,layer in enumerate(layers):
            #if layer.type() == "RasterLayer":
            path = str(layer.dataProvider().dataSourceUri()).replace("|layerid=0","")
            comboBox.addItem(_fromUtf8(""))
            comboBox.setItemText(i+1, _translate("Pansharp", path, None))

    # run method that performs all the real work
    def pansharp(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_pansharp = PansharpDialog()
        self.changeActive(self.dlg_pansharp.ui.comboBox_multiband)
        self.changeActive(self.dlg_pansharp.ui.comboBox_panchromatic)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_pansharp.ui.comboBox_multiband))
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_pansharp.ui.comboBox_panchromatic))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_pansharp.show()
        # Run the dialog event loop
        result = self.dlg_pansharp.exec_()
        if result == 1:
            ui = self.dlg_pansharp.ui
            dlgProgress.show()
            multiband_image = parse_input(str(ui.comboBox_multiband.currentText()))
            panchromatic_image = parse_input(str(ui.comboBox_panchromatic.currentText()))
            output_image = str(ui.lineEdit_output.text())
            rowsp,colsp,nbands,geo_transform,projection = read_image_parameters(panchromatic_image)
            rowsxs,colsxs,nbands,geo_transform,projection = read_image_parameters(multiband_image)
            scale_rows = round(float(rowsp)/float(rowsxs),0)
            scale_cols = round(float(colsp)/float(colsxs),0)
            executeOtb("otbcli_RigidTransformResample -progress 1 -in {} -out {} -transform.type id -transform.type.id.scalex {} -transform.type.id.scaley {}".format(multiband_image,multiband_image[:-4]+'_resampled.tif',scale_cols,scale_rows),progress=dlgProgress.ui,label="Resampling")
            fix_tiling_raster(panchromatic_image,multiband_image[:-4]+'_resampled.tif')
            executeOtb("otbcli_Pansharpening -progress 1 -inp {} -inxs {} -out {} uint16".format(panchromatic_image,multiband_image[:-4]+'_resampled.tif',output_image),progress=dlgProgress.ui,label="Pan-sharpening")
            QgsMapLayerRegistry.instance().addMapLayer(QgsRasterLayer(output_image, QFileInfo(output_image).baseName()))
            QMessageBox.information(None, "Info", 'Done!')

    def classification(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_classification = ClassificationDialog()
        self.changeActive(self.dlg_classification.ui.comboBox_input)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_classification.ui.comboBox_input))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_classification.show()
        # Run the dialog event loop
        result = self.dlg_classification.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_classification.ui
            dlgProgress.show()
            input_image = parse_input(str(ui.comboBox_input.currentText()))
            output_raster = str(ui.lineEdit_output.text())
            input_classification_type = str(ui.comboBox_supervised.currentText())
            if input_classification_type == "Supervised":
                input_shape_list = [str(ui.lineEdit_training.text())]
                input_classification_supervised_type = str(ui.comboBox_supervised_type.currentText())
                training_field = str(ui.lineEdit_training_field.text())
                input_raster_list = [input_image]
                input_raster = input_image
                root = ET.Element("FeatureStatistics")
                print 'Number of provided raster files: ' + str(len(input_raster_list))
                for i in range(len(input_raster_list)):
                    rows,cols,nbands,geotransform,projection = read_image_parameters(input_raster_list[i])
                    band_list = read_image(input_raster_list[i],np.uint16,0)
                    statistic = ET.SubElement(root,"Statistic")
                    statistic.set("name","mean")
                    for b in range(nbands):
                        statistic_vector = ET.SubElement(statistic,"StatisticVector")
                        statistic_vector.set("value",str(round(np.mean(band_list[b]),4)))
                for i in range(len(input_raster_list)):
                    band_list = read_image(input_raster_list[i],np.uint16,0)
                    statistic = ET.SubElement(root,"Statistic")
                    statistic.set("name","stddev")
                    for b in range(nbands):
                        statistic_vector = ET.SubElement(statistic,"StatisticVector")
                        statistic_vector.set("value",str(round(np.std(band_list[b])/2,4)))
                tree = ET.ElementTree(root)
                tree.write(input_raster_list[0][:-4]+'_statistics.xml')
                input_text = "tmp.txt"
                executeOtb("otbcl i_TrainImagesClassifier -progress 1 -io.il {} -io.vd {} -io.imstat {} -sample.mv 100 -sample.mt 100 -sample.vtr 0.5 -sample.edg 1 -sample.vfn {} -classifier {} -io.out {} -io.confmatout {} ".format(input_raster_list[0],input_shape_list[0],input_raster_list[0][:-4]+'_statistics.xml',training_field,input_classification_supervised_type,input_text,input_text[:-4] + "_ConfusionMatrix.csv"),progress=dlgProgress.ui,label="Training Classifier")
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
                executeOtb("otbcli_ImageClassifier -progress 1 -in {} -imstat {} -model {} -out {}".format(input_raster,input_raster[:-4]+'_statistics.xml',input_text,output_raster),progress=dlgProgress.ui,label="Classification")
            else:
                n_classes = int(ui.spinBox_nclasses.text())
                n_iterations = int(ui.spinBox_niteration.text())
                executeOtb("otbcli_KMeansClassification -progress 1 -in {} -ts 1000 -nc {} -maxit {} -ct 0.0001 -out {}".format( input_image, n_classes, n_iterations, output_raster),progress=dlgProgress.ui,label="Unsupervised classification")
            QgsMapLayerRegistry.instance().addMapLayer(QgsRasterLayer(output_raster, QFileInfo(output_raster).baseName()))
            QMessageBox.information(None, "Info", 'Done!')

    def segmentation(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_segmantation = SegmentationDialog()
        self.changeActive(self.dlg_segmantation.ui.comboBox_input)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_segmantation.ui.comboBox_input))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_segmantation.show()
        # Run the dialog event loop
        result = self.dlg_segmantation.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_segmantation.ui
            dlgProgress.show()
            input_image = parse_input(str(ui.comboBox_input.currentText()))
            output_shape = str(ui.lineEdit_output.text())
            checked_optimizer = bool(ui.checkBox_optimizer.isChecked())
            segm_mode = str(ui.comboBox_method.currentText())
            if checked_optimizer:
                optimizer_shape = str(ui.lineEdit_optimizer_input.text())
                nloops = int(ui.spinBox_nloops.text())
                select_criteria = int(ui.spinBox_criteria.text())
                floaters = bool(ui.radioButton_floaters.isChecked())
                segm_mode_optimizer = segm_mode
                if floaters and segm_mode == "Baatz":
                    segm_mode_optimizer = "Baatz"
                elif floaters == 0 and segm_mode == "Baatz":
                    segm_mode_optimizer = "Baatz_integers"
                elif floaters and segm_mode == "Region Growing":
                    segm_mode_optimizer = "Region_growing"
                elif floaters == 0 and segm_mode == "Region Growing":
                    segm_mode_optimizer = "Region_growing_integers"
                elif segm_mode == "Morphological Profiles":
                    segm_mode_optimizer = "Mprofiles"
                band_list = read_image(input_image,np.uint16,0)
                rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                #Open reference shapefile
                driver_shape = osgeo.ogr.GetDriverByName('ESRI Shapefile')
                inDS = driver_shape.Open(optimizer_shape, 0)
                if inDS is None:
                    print 'Could not open file'
                    sys.exit(1)
                inLayer = inDS.GetLayer()
                numFeatures = inLayer.GetFeatureCount()
                print 'Number of reference features: ' + str(numFeatures)
                patches_list = []
                patches_geo_transform_list = []
                reference_list = []
                ref_geo_transform_list = []
                dlgProgress.ui.progressBar.setMinimum(1)
                dlgProgress.ui.progressBar.setMaximum(numFeatures)
                for n in range(0,numFeatures):
                    #separate each polygon creating a temp file
                    temp = split_shape(inLayer,n)
                    #conversion of the temp file to raster
                    temp_layer = temp.GetLayer()
                    reference_matrix, ref_geo_transform = polygon2array(temp_layer,geo_transform[1],abs(geo_transform[5])) 
                    temp.Destroy()
                    reference_list.append(reference_matrix)
                    ref_geo_transform_list.append(ref_geo_transform)
                    ext_patch_list,patch_geo_transform = create_extended_patch(band_list,reference_matrix,geo_transform,ref_geo_transform,0.3,False)
                    patches_list.append(ext_patch_list)
                    patches_geo_transform_list.append(patch_geo_transform)
                    dlgProgress.ui.progressBar.setValue(dlgProgress.ui.progressBar.value()+1)
                e = call_optimizer(segm_mode_optimizer,patches_list,reference_list,patches_geo_transform_list,ref_geo_transform_list,projection,select_criteria,nloops)
                if segm_mode_optimizer == 'Felzenszwalb':
                    input_band_list = read_image(input_image,0,0)
                    rows_fz,cols_fz,nbands_fz,geotransform_fz,projection_fz = read_image_parameters(input_image)
                    segments_fz = felzenszwalb_skimage(input_band_list, float(e[0]), float(e[1]), 0)
                    write_image([segments_fz],0,0,output_shape[:-4]+'.TIF',rows_fz,cols_fz,geotransform_fz,projection_fz)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
                if segm_mode_optimizer == 'Edison':
                    paramaters = {"spatialr": round(e[0]), "ranger": round(e[1]), "minsize": 0, "scale": 0}
                    QMessageBox.information(None, "Info", paramaters)
                    segmentation_progress(input_image, output_shape, "edison", paramaters=paramaters,progress=dlgProgress.ui)
                    #edison_otb(input_image,"vector",output_shape,int(round(e[0])),int(round(e[1])),0,0)
                if segm_mode_optimizer == 'Meanshift':
                    paramaters = {"spatialr": round(e[0]), "ranger": round(e[1]), "thres": 0, "maxiter": 0,"minsize": 0}
                    segmentation_progress(input_image, output_shape, "meanshift", paramaters=paramaters,progress=dlgProgress.ui)
                    #meanshift_otb(input_image,output_shape,'vector',int(round(e[0])),float(e[1]),0,0,0)
                if segm_mode_optimizer == 'Watershed':
                    paramaters = {"threshold": 0, "level": float(e[0])}
                    segmentation_progress(input_image, output_shape, "watershed", paramaters=paramaters,progress=dlgProgress.ui)
                    #watershed_otb(input_image,'vector',output_shape,0,float(e[0]))
                if segm_mode_optimizer == 'Mprofiles':
                    mprofiles_otb(input_image,output_shape,'vector',0,int(round(e[0])),0,0)
                if segm_mode == 'Baatz':
                    if floaters:
                        segments_baatz = baatz_interimage(input_image,0,float(e[0]),float(e[1]),0,True)
                    else:    
                        segments_baatz = baatz_interimage(input_image,int(round(e[0])),Compact,Color,int(round(e[1])),True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_baatz],0,0,output_shape[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
                if segm_mode == 'Region_growing':
                    if floaters:
                        segments_regiongrowing = region_growing_interimage(input_image,int(round(e[0])),0,0,int(round(e[1])),True)
                    else:
                        segments_regiongrowing = region_growing_interimage(input_image,EuclideanT,float(e[0]),float(e[1]),Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_regiongrowing],0,0,output_shape[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
            else:
                if segm_mode == "Felzenszwalb":
                    min_size = float(ui.lineEdit_felzenszwalb_minsize.text())
                    scale = float(ui.lineEdit_felzenszwalb_scale.text())
                    sigma = int(ui.lineEdit_felzenszwalb_sigma.text())
                    input_band_list = read_image(input_image,0,0)
                    rows_fz,cols_fz,nbands_fz,geotransform_fz,projection_fz = read_image_parameters(input_image)
                    segments_fz = felzenszwalb_skimage(input_band_list, scale, sigma, min_size)
                    write_image([segments_fz],0,0,output_shape[:-4]+'.TIF',rows_fz,cols_fz,geotransform_fz,projection_fz)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
                if segm_mode == "Edison":
                    spatial_radius = int(ui.lineEdit_edison_radius.text())
                    range_radius = int(ui.lineEdit_edison_range.text())
                    min_size = int(ui.lineEdit_edison_size.text())
                    scale = int(ui.lineEdit_edison_scale.text())
                    paramaters = {"spatialr": spatial_radius, "ranger": range_radius, "minsize": min_size, "scale": scale}
                    segmentation_progress(input_image, output_shape, "edison", paramaters=paramaters,progress=dlgProgress.ui)
                elif segm_mode == "Baatz":
                    EuclideanT = int(ui.lineEdit_baatz_euclidean.text())
                    Compact = float(ui.lineEdit_baatz_compactness.text())
                    Color = float(ui.lineEdit_baatz_color.text())
                    Scale = int(ui.lineEdit_baatz_scale.text())
                    segments_baatz = baatz_interimage(input_image,EuclideanT,Compact,Color,Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_baatz],0,0,output_shape[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
                elif segm_mode == "Meanshift":
                    SpatialR = int(ui.lineEdit_meanshift_spatial.text())
                    RangeR = float(ui.lineEdit_meanshift_range.text())
                    Thres = float(ui.lineEdit_meanshift_threshold.text())
                    MaxIter = int(ui.lineEdit_meanshift_iterations.text())
                    MinimumS = int(ui.lineEdit_meanshift_minsize.text())
                    paramaters = {"spatialr": SpatialR, "ranger": RangeR, "thres": Thres, "maxiter": MaxIter,"minsize": MinimumS}
                    segmentation_progress(input_image, output_shape, "meanshift", paramaters=paramaters,progress=dlgProgress.ui)
                elif segm_mode == "Watershed":
                    Thres = float(ui.lineEdit_watershed_threshold.text())
                    Level = float(ui.lineEdit_watershed_level.text())
                    paramaters = {"threshold": Thres, "level": Level}
                    segmentation_progress(input_image, output_shape, "watershed", paramaters=paramaters,progress=dlgProgress.ui)
                elif segm_mode == "Morphological Profiles":
                    Size = int(ui.lineEdit_morphological_radius.text())
                    Sigma = float(ui.lineEdit_morphological_sigma.text())
                    Start = float(ui.lineEdit_morphological_start.text())
                    Step = int(ui.lineEdit_morphological_step.text())
                    paramaters = {"size": Size, "sigma": Sigma, "start": Start, "step": Step}
                    segmentation_progress(input_image, output_shape, "mprofiles", paramaters=paramaters,progress=dlgProgress.ui)
                elif segm_mode == "Region Growing":
                    EuclideanT = int(ui.lineEdit_region_euclidean.text())
                    Compact = float(ui.lineEdit_region_compactness.text())
                    Color = float(ui.lineEdit_region_color.text())
                    Scale = int(ui.lineEdit_region_scale.text())
                    segments_regiongrowing = region_growing_interimage(input_image,EuclideanT,Compact,Color,Scale,True)
                    rows,cols,nbands,geo_transform,projection = read_image_parameters(input_image)
                    write_image([segments_regiongrowing],0,0,output_shape[:-4]+'.TIF',rows,cols,geo_transform,projection)
                    rast2shp(output_shape[:-4]+'.TIF',output_shape)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def features(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_features = FeaturesDialog()
        self.changeActive(self.dlg_features.ui.comboBox_input)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_features.ui.comboBox_input))
        self.changeActive(self.dlg_features.ui.comboBox_training)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_features.ui.comboBox_training))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_features.show()
        # Run the dialog event loop
        result = self.dlg_features.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_features.ui
            dlgProgress.show()
            input_image = parse_input(str(ui.comboBox_input.currentText()))
            input_shape = parse_input(str(ui.comboBox_training.currentText()))
            output_shape = str(ui.lineEdit_output.text())
            field = str(ui.lineEdit_field.text())
            spectrals = [(ui.checkBox_multiprocess,"--multi"),(ui.checkBox_maxbr,"--max_br"),(ui.checkBox_mean,"--mean"),(ui.checkBox_minbr,"--min_br"),(ui.checkBox_mode,"--mode"),(ui.checkBox_ndivistd,"--ndvi_std"),(ui.checkBox_ndvimean,"--ndvi_mean"),(ui.checkBox_std,"--std"),(ui.checkBox_weighbr,"--weigh_br")]
            textures = [(ui.checkBox_asm,"--ASM"),(ui.checkBox_contrast,"--contrast"),(ui.checkBox_correlation,"--correlation"),(ui.checkBox_dissimilarity,"--dissimilarity"),(ui.checkBox_energy,"--energy"),(ui.checkBox_homogeneity,"--homogeneity")]
            indexes_list_spectral = " ".join([index for pushButton,index in spectrals if pushButton.isChecked()])
            indexes_list_texture = " ".join([index for pushButton,index in textures if pushButton.isChecked()])
            executeScript('/scripts/test_features.py" \"{}\" \"{}\" \"{}\" \"{}\" {} {}'.format(input_image,input_shape,output_shape,field,indexes_list_spectral,indexes_list_texture),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def build_height(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_height = BuildHeightDialog()
        self.changeActive(self.dlg_height.ui.comboBox_input_buildings)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_height.ui.comboBox_input_buildings))
        self.changeActive(self.dlg_height.ui.comboBox_input_shadows)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_height.ui.comboBox_input_shadows))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_height.show()
        # Run the dialog event loop
        result = self.dlg_height.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_height.ui
            dlgProgress.show()
            input_buildings = parse_input(str(ui.comboBox_input_buildings.currentText()))
            input_shadow = parse_input(str(ui.comboBox_input_shadows.currentText()))
            output_shape = str(ui.lineEdit_output.text())
            idfield = str(ui.lineEdit_shadow_field.text())
            window_resize = str(ui.doubleSpinBox_window_paramater.text()).replace(",",".")
            window_resize = float(window_resize)
            date = str(ui.dateTimeEdit.text())
            executeScript('/scripts/height.py" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"'.format(input_shadow,input_buildings,date,output_shape,idfield,window_resize),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def coregistration(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_coregistration = CoregistrationDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_coregistration.show()
        # Run the dialog event loop
        result = self.dlg_coregistration.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_coregistration.ui
            dlgProgress.show()
            target_folder = str(ui.lineEdit_tobechange.text())
            reference_path = str(ui.lineEdit_reference.text())
            options = [(ui.checkBox_resampling , "--enable_resampling"),(ui.checkBox_surf, "--enable_SURF" ),(ui.checkBox_fft, "--enable_FFT")]
            options = " ".join([index for pushButton,index in options if pushButton.isChecked()])
            select_crop = ("--enable_clip {} ".format(string_qmark(parse_input(ui.comboBox_input_shape.currentText()))) if str(ui.comboBox_select_crop.currentText()) == "Clip" else "--enable_grid {} {} ".format(ui.spinBox_rows.text(), ui.spinBox_cols.text()))
            executeScript('/scripts/coregistration.py" \"{}\" \"{}\" {} {}'.format(reference_path,target_folder,select_crop,options),dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')


    def footprints(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_footprints = FootprintsDialog()
        self.changeActive(self.dlg_footprints.ui.comboBox_pansharp)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_footprints.ui.comboBox_pansharp))
        self.changeActive(self.dlg_footprints.ui.comboBox_training)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_footprints.ui.comboBox_training))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_footprints.show()
        # Run the dialog event loop
        result = self.dlg_footprints.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_footprints.ui
            dlgProgress.show()
            pansharp_file = parse_input(str(ui.comboBox_pansharp.currentText()))
            training_set = parse_input(str(ui.comboBox_training.currentText()))
            training_attribute = str(ui.lineEdit_training_field.text())
            output_shape = str(ui.lineEdit_output.text())
            checked_optimizer = ("--optimazer" if bool(ui.checkBox_filter.isChecked()) else "")
            if checked_optimizer == "--optimazer":
                executeOtb("otbcli_MeanShiftSmoothing -progress 1 -in {} -spatialr {} -ranger {} -thres {} -maxiter {} -fout {} -foutpos {}".format(pansharp_file,30,30,0.1,100,pansharp_file[:-4]+'_smooth.tif',pansharp_file[:-4]+'_sp.tif'),progress=dlgProgress.ui,label="Smooth Filter")
                pansharp_file = pansharp_file[:-4]+'_smooth.tif'
                checked_optimizer = ""
            building_classes = [str(ui.listWidget.item(index).text()) for index in xrange(ui.listWidget.count())]
            building_classes = ("-c \""+"\" \"".join(building_classes)+"\"" if len(building_classes) else "")
            executeScript('/scripts/footprints.py" \"{}\" \"{}\" \"{}\" \"{}\" {} {}'.format(pansharp_file,training_set,training_attribute,output_shape,checked_optimizer,building_classes),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(output_shape,os.path.splitext(os.path.basename(output_shape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')

    def stacksatellite(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_stacksatellite = StackSatelliteDialog()
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_stacksatellite.show()
        # Run the dialog event loop
        result = self.dlg_stacksatellite.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_stacksatellite.ui
            dlgProgress.show()
            sat_folder = str(ui.lineEdit_satellite_folder.text())
            segmentation_name = str(ui.comboBox_segmentation.currentText())
            n_classes = str(ui.spinBox_nclasses.text())
            ref_dir = ("--ref_dir "+string_qmark(str(ui.lineEdit_reference_directory.text())) if bool(ui.checkBox_reference_diretory.isChecked()) else "")
            input_shapefile = ("--restrict_to_city "+string_qmark(parse_input(str(ui.comboBox_input_shapefile.currentText()))) if bool(ui.checkBox_restrict_to_city.isChecked()) else "")
            options = [(ui.checkBox_coregistration , "--coregistration"),(ui.checkBox_builtup_index, "--builtup_index_method" ),(ui.checkBox_pca_index, "--pca_index_method"),(ui.checkBox_pca_classification , "--pca_classification_method"),(ui.checkBox_dissimilarity, "--dissimilarity_method" ),(ui.checkBox_pca_ob, "--pca_ob_method" )]
            options = " ".join([index for pushButton,index in options if pushButton.isChecked()])
            executeScript('/scripts/stacksatellite.py" \"{}\" \"{}\" \"{}\" {} {} {}'.format(sat_folder,segmentation_name,n_classes,ref_dir,input_shapefile,options),dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')

    def density(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_density = DensityDialog()
        self.changeActive(self.dlg_density.ui.comboBox_building_shape)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_density.ui.comboBox_building_shape))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_density.show()
        # Run the dialog event loop
        result = self.dlg_density.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_density.ui
            dlgProgress.show()
            buildingShape = parse_input(str(ui.comboBox_building_shape.currentText()))
            radius = str(ui.doubleSpinBox_radius.text()).replace(",",".")
            radius = float(radius)
            outputShape = str(ui.lineEdit_output_shapefile.text())
            if os.path.isfile(outputShape): os.remove(outputShape)
            executeScript('/scripts/density.py" \"{}\" \"{}\" \"{}\"'.format(buildingShape,radius,outputShape),dlgProgress.ui)
            QgsMapLayerRegistry.instance().addMapLayer(QgsVectorLayer(outputShape,os.path.splitext(os.path.basename(outputShape))[0], "ogr"))
            QMessageBox.information(None, "Info", 'Done!')
    
    def temporal(self):
        # Create the dialog (after translation) and keep reference
        self.dlg_temporal = TemporalDialog()
        self.changeActive(self.dlg_temporal.ui.comboBox_mask)
        QObject.connect(self.iface.mapCanvas(), SIGNAL( "layersChanged()" ), lambda: self.changeActive(self.dlg_temporal.ui.comboBox_mask))
        dlgProgress = ProgressDialog()
        # show the dialog
        self.dlg_temporal.show()
        # Run the dialog event loop
        result = self.dlg_temporal.exec_()
        # See if OK was pressed
        if result == 1:
            ui = self.dlg_temporal.ui
            dlgProgress.show()
            sat_folder = str(ui.lineEdit_folder.text())
            input_mask = parse_input(str(ui.comboBox_mask.currentText()))
            n_classes = str(ui.spinBox_nclass.text())
            indexes = [(ui.checkBox_1,"Index1"),(ui.checkBox_2,"Index2"),(ui.checkBox_3,"Index3"),(ui.checkBox_4,"Index4"),(ui.checkBox_5,"Index5"),(ui.checkBox_6,"Index6"),(ui.checkBox_7,"Index7"),(ui.checkBox_8,"Index8"),(ui.checkBox_9,"Index9"),(ui.checkBox_10,"Index10"),(ui.checkBox_11,"Index11"),(ui.checkBox_12,"Index12")]
            indexes_list = " ".join([index for pushButton,index in indexes if pushButton.isChecked()])
            executeScript('/scripts/temporal.py" \"{}\" \"{}\" \"{}\" -i {}'.format(sat_folder,input_mask,n_classes,indexes_list),dlgProgress.ui)
            QMessageBox.information(None, "Info", 'Done!')
